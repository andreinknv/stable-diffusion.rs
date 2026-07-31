//! Circular padding, which is what makes an image tile.
//!
//! Exact arithmetic on a tiny array, because the end-to-end test can only
//! observe the *consequence* — and observing it needs a picture whose borders
//! carry detail, which is a property of the prompt and the step count rather
//! than of the padding.
#![cfg(feature = "mlx")]

use sd_tensor::mlx::{Array, Stream};

/// **The pad comes from the opposite edge**, in the right order.
///
/// Getting head and tail the wrong way round still produces an array of the
/// right shape whose edges still "match" under a symmetry test — it mirrors
/// instead of wrapping — so the check is against the exact values.
#[test]
fn padding_wraps_from_the_far_edge() {
    let s = Stream::cpu();
    // One row, four columns, one channel: [1, 1, 4, 1] in NHWC.
    let x = Array::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4, 1]).unwrap();
    let p = x.pad_circular(&[2], 1, &s).unwrap();
    assert_eq!(p.shape(), vec![1, 1, 6, 1]);
    assert_eq!(
        p.to_vec_f32(&s).unwrap(),
        vec![4.0, 1.0, 2.0, 3.0, 4.0, 1.0],
        "the left pad is the last column and the right pad is the first"
    );
}

/// Both spatial axes, together, as a convolution asks for them.
#[test]
fn padding_wraps_on_every_axis_asked_for() {
    let s = Stream::cpu();
    // [1, 2, 2, 1]:  1 2
    //                3 4
    let x = Array::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2, 1]).unwrap();
    let p = x.pad_circular(&[1, 2], 1, &s).unwrap();
    assert_eq!(p.shape(), vec![1, 4, 4, 1]);
    // Wrapping both ways makes the 2x2 tile repeat, with the centre offset by
    // one — every corner is the diagonally opposite value.
    assert_eq!(
        p.to_vec_f32(&s).unwrap(),
        vec![
            4.0, 3.0, 4.0, 3.0, //
            2.0, 1.0, 2.0, 1.0, //
            4.0, 3.0, 4.0, 3.0, //
            2.0, 1.0, 2.0, 1.0,
        ]
    );
}

/// Zero is the identity, so a convolution with no padding is unaffected.
#[test]
fn padding_by_zero_changes_nothing() {
    let s = Stream::cpu();
    let x = Array::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4, 1]).unwrap();
    let p = x.pad_circular(&[1, 2], 0, &s).unwrap();
    assert_eq!(p.shape(), x.shape());
    assert_eq!(p.to_vec_f32(&s).unwrap(), x.to_vec_f32(&s).unwrap());
}

/// **Wrapping by more than the axis is long is refused.**
///
/// It would need the input repeated, which no convolution here asks for; doing
/// it silently would produce a tiled input rather than a padded one, at the
/// right shape.
#[test]
fn wrapping_further_than_the_axis_is_refused() {
    let s = Stream::cpu();
    let x = Array::from_slice_f32(&[1.0, 2.0], &[1, 1, 2, 1]).unwrap();
    assert!(x.pad_circular(&[2], 3, &s).is_err());
    // And exactly the length is still fine.
    assert!(x.pad_circular(&[2], 2, &s).is_ok());
}
