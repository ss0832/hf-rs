// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! Allocation regression for ECP quadrature. Run separately so unrelated tests cannot affect the peak.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use hf_rs::Atom;
use hf_rs::basis::{ContractedGaussian, Primitive};
use hf_rs::ecp::{EcpDef, EcpQuadrature, EcpTerm, ecp_one_electron_matrix_with};

struct CountingAllocator;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn allocated(bytes: usize) {
    let live = LIVE.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK.fetch_max(live, Ordering::Relaxed);
}

// This allocator delegates to System and only records successful allocations.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            allocated(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            allocated(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            if new_size >= layout.size() {
                allocated(new_size - layout.size());
            } else {
                LIVE.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        new_ptr
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn dense_angular_quadrature_has_bounded_workspace_and_preserves_projector_integrals() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    pool.install(|| {});
    let term = |exponent, coefficient| EcpTerm {
        power: 2,
        exponent,
        coefficient,
    };
    let mut atom = Atom::new("H", [0.0; 3]);
    atom.ecp = Some(Arc::new(EcpDef {
        symbol: "H".into(),
        n_core: 0,
        l_max: 1,
        local: vec![term(0.9, -1.7)],
        semilocal: vec![vec![term(0.6, 2.3)]],
    }));
    let basis: Vec<_> = (0..32)
        .map(|i| ContractedGaussian {
            atom_index: 0,
            center: [0.0; 3],
            l: [0; 3],
            primitives: vec![Primitive {
                exponent: 0.5 + 0.01 * i as f64,
                coefficient: 1.0,
            }],
        })
        .collect();
    // About 20,000 points/shell: the old full-grid matrices exceed 32 MiB with four workers.
    let q = EcpQuadrature {
        panel_nodes: 24,
        panel_ratio: 2.0,
        first_edge: 1.0,
        angular_margin: 196,
        ..EcpQuadrature::default()
    };
    let baseline = LIVE.load(Ordering::Relaxed);
    PEAK.store(baseline, Ordering::Relaxed);
    let m = pool.install(|| ecp_one_electron_matrix_with(&[atom], &basis, &q));
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(baseline);
    println!(
        "ECP workspace peak above baseline: {:.3} MiB",
        peak as f64 / 1048576.0
    );
    assert!(
        peak < 32 * 1024 * 1024,
        "ECP quadrature allocated {} bytes",
        peak
    );
    for i in 0..basis.len() {
        for j in 0..basis.len() {
            let ab = basis[i].primitives[0].exponent + basis[j].primitives[0].exponent;
            // Every centered s function lies wholly in P_0, including cross-block angular terms.
            let exact = std::f64::consts::PI.powf(1.5)
                * (-1.7 / (ab + 0.9).powf(1.5) + 2.3 / (ab + 0.6).powf(1.5));
            assert!(
                (m[(i, j)] - exact).abs() < 1e-11,
                "({}, {}): {} vs {}",
                i,
                j,
                m[(i, j)],
                exact
            );
        }
    }
}
