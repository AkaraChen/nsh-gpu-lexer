//! The transcendental functions of the JS reference, bit for bit.
//!
//! V8 implements `Math.exp` and `Math.tanh` with fdlibm (FreeBSD msun, in
//! `base/ieee754.cc`). The `libm` crate ports the same msun `exp` and
//! `expm1`; `tanh` is ported here from msun `s_tanh.c` because `libm`'s
//! follows musl's different formula. Using these instead of the platform's
//! libm keeps every target (glibc, Apple, MSVC, wasm) on the same results.

/// `Math.exp`.
#[inline]
pub fn exp(x: f64) -> f64 {
    libm::exp(x)
}

/// `Math.tanh`, msun `s_tanh.c`.
#[inline]
pub fn tanh(x: f64) -> f64 {
    const ONE: f64 = 1.0;
    const TWO: f64 = 2.0;
    const TINY: f64 = 1.0e-300;
    const HUGE: f64 = 1.0e300;
    let jx = (x.to_bits() >> 32) as u32 as i32;
    let ix = jx & 0x7fff_ffff;
    // x is INF or NaN.
    if ix >= 0x7ff0_0000 {
        return if jx >= 0 { ONE / x + ONE } else { ONE / x - ONE };
    }
    let z;
    if ix < 0x4036_0000 {
        // |x| < 22
        if ix < 0x3e30_0000 {
            // |x| < 2**-28: tanh(tiny) = tiny
            if HUGE + x > ONE {
                return x;
            }
        }
        if ix >= 0x3ff0_0000 {
            // |x| >= 1
            let t = libm::expm1(TWO * x.abs());
            z = ONE - TWO / (t + TWO);
        } else {
            let t = libm::expm1(-TWO * x.abs());
            z = -t / (t + TWO);
        }
    } else {
        // |x| >= 22: +-1
        z = ONE - TINY;
    }
    if jx >= 0 { z } else { -z }
}

/// `1 / (1 + Math.exp(-x))`, as the reference writes its sigmoids.
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + exp(-x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tanh_matches_std_closely() {
        for i in -4000..4000 {
            let x = f64::from(i) * 0.01;
            let ours = tanh(x);
            let theirs = x.tanh();
            assert!(
                (ours - theirs).abs() <= 2.0 * f64::EPSILON * theirs.abs().max(1e-300),
                "{x}"
            );
        }
        assert_eq!(tanh(0.0), 0.0);
        assert_eq!(tanh(100.0), 1.0);
        assert_eq!(tanh(-100.0), -1.0);
    }
}
