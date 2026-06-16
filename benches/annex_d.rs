//! Criterion benchmarks for the H.261 Annex D still-image
//! sub-sample / re-assembly transforms.
//!
//! Round 320 (depth-mode benchmark). The existing benches cover the
//! per-block transform (`transform`), the per-block P-picture
//! reconstruction primitives (`filter_mc`), the BCH §5.4 FEC layer
//! (`bch`), the §4 start-code scanner (`start_code`), and the
//! end-to-end encode / decode picture cost (`encode` / `decode`).
//!
//! What was missing was an A/B baseline for the **Annex D §D.2 /
//! Figure D.1** still-image transform. When an H.261 endpoint sends a
//! still image (HI_RES = 0), it 2:1×2:1 sub-samples a full-resolution
//! still image (4× the video format) into four sub-images, each at the
//! current video format, and the receiver re-assembles them. Unlike
//! the 8×8 block primitives, these run over a *whole plane*: a QCIF
//! endpoint's still image is a 352×288 CIF luma frame plus 176×144
//! chroma; a CIF endpoint's still image is a 704×576 CCIR-601 luma
//! frame plus 352×288 chroma. The interleaved strided read
//! (`subsample_plane`: gather every other sample) and the scatter write
//! (`reassemble_plane`: place each sub-image sample back at `2x+dx,
//! 2y+dy`) are memory-bandwidth-bound and a natural target for a future
//! cache-blocked or SIMD-gather optimisation — but had no baseline.
//!
//! Both `subsample_plane` / `reassemble_plane` are pure public
//! functions, as are their YUV-4:2:0 wrappers `subsample_still_image` /
//! `reassemble_still_image`. This bench gives all four an A/B baseline
//! distinct from the per-block numbers.
//!
//! Throughput is reported in bytes (still-image plane size) so the
//! per-sample cost lands as a bandwidth figure in criterion's report.
//!
//! Run with:
//!     cargo bench -p oxideav-h261 --bench annex_d

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use oxideav_h261::annex_d::{
    reassemble_plane, reassemble_still_image, still_image_chroma_dimensions,
    still_image_dimensions, subsample_plane, subsample_still_image, SubImagePlanes,
};
use oxideav_h261::picture::SourceFormat;

/// Cheap deterministic LCG — fills the still-image planes with content
/// that varies per sample so the strided gather / scatter never
/// degenerates to a memset the optimiser could elide. (A flat plane
/// would let the compiler hoist the load; we want every sample
/// distinct.)
fn fill_plane(len: usize, seed: u32) -> Vec<u8> {
    let mut s = seed;
    let mut out = vec![0u8; len];
    for b in out.iter_mut() {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        *b = (s >> 16) as u8;
    }
    out
}

/// Build the three full-resolution still-image planes (Y, Cb, Cr) for
/// a given currently-transmitted video format.
fn build_still_planes(fmt: SourceFormat) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (sw, sh) = still_image_dimensions(fmt);
    let (cw, ch) = still_image_chroma_dimensions(fmt);
    let y = fill_plane((sw * sh) as usize, 0x1234_5678);
    let cb = fill_plane((cw * ch) as usize, 0x9ABC_DEF0);
    let cr = fill_plane((cw * ch) as usize, 0x0F0F_0F0F);
    (y, cb, cr)
}

fn bench_subsample_plane(c: &mut Criterion) {
    let mut group = c.benchmark_group("annex_d_subsample_plane");

    for fmt in [SourceFormat::Qcif, SourceFormat::Cif] {
        let (sw, sh) = still_image_dimensions(fmt);
        let plane = fill_plane((sw * sh) as usize, 0xCAFE_F00D);
        let label = match fmt {
            SourceFormat::Qcif => "qcif_still_352x288",
            SourceFormat::Cif => "cif_still_704x576",
        };
        group.throughput(Throughput::Bytes((sw * sh) as u64));
        group.bench_function(BenchmarkId::new("subsample_plane", label), |b| {
            b.iter(|| {
                let out = subsample_plane(black_box(&plane), sw, sh);
                black_box(out);
            });
        });
    }

    group.finish();
}

fn bench_reassemble_plane(c: &mut Criterion) {
    let mut group = c.benchmark_group("annex_d_reassemble_plane");

    for fmt in [SourceFormat::Qcif, SourceFormat::Cif] {
        let (sw, sh) = still_image_dimensions(fmt);
        let plane = fill_plane((sw * sh) as usize, 0xCAFE_F00D);
        // The reassemble input is the sub-sample output — build it once
        // so the bench times only the scatter write.
        let subs = subsample_plane(&plane, sw, sh);
        let label = match fmt {
            SourceFormat::Qcif => "qcif_still_352x288",
            SourceFormat::Cif => "cif_still_704x576",
        };
        group.throughput(Throughput::Bytes((sw * sh) as u64));
        group.bench_function(BenchmarkId::new("reassemble_plane", label), |b| {
            b.iter(|| {
                let out = reassemble_plane(black_box(&subs), sw, sh);
                black_box(out);
            });
        });
    }

    group.finish();
}

fn bench_still_image_yuv(c: &mut Criterion) {
    let mut group = c.benchmark_group("annex_d_still_image_yuv");

    for fmt in [SourceFormat::Qcif, SourceFormat::Cif] {
        let (sw, sh) = still_image_dimensions(fmt);
        let (cw, ch) = still_image_chroma_dimensions(fmt);
        // Aggregate throughput = all three planes' bytes (the full
        // YUV-4:2:0 still image the §D.2 transform touches per call).
        let total = (sw * sh + 2 * cw * ch) as u64;
        let (y, cb, cr) = build_still_planes(fmt);
        let label = match fmt {
            SourceFormat::Qcif => "qcif_still",
            SourceFormat::Cif => "cif_still",
        };

        group.throughput(Throughput::Bytes(total));
        group.bench_function(BenchmarkId::new("subsample_still_image", label), |b| {
            b.iter(|| {
                let subs =
                    subsample_still_image(fmt, black_box(&y), black_box(&cb), black_box(&cr));
                black_box(subs);
            });
        });

        // Build the four sub-image planes once for the re-assembly bench
        // so it times only the inverse transform.
        let subs: [SubImagePlanes; 4] = subsample_still_image(fmt, &y, &cb, &cr);
        group.throughput(Throughput::Bytes(total));
        group.bench_function(BenchmarkId::new("reassemble_still_image", label), |b| {
            b.iter(|| {
                let recon = reassemble_still_image(fmt, black_box(&subs));
                black_box(recon);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_subsample_plane,
    bench_reassemble_plane,
    bench_still_image_yuv
);
criterion_main!(benches);
