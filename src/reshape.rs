use itertools::Itertools;
use nncombinator::arr::Arr2;
use nncombinator::mem::AsRawSlice;
use crate::{Assert, assert_convolution, IsTrue};

use crate::collection::{ImageView};

pub fn im2col<'a,T,const H:usize,const W:usize,const FH:usize,const FW:usize,const PAD:usize,const S:usize>(image:&ImageView<'a,T,H,W>)
    -> Arr2<T,{ ((H + PAD * 2 - FH) / S + 1) * ((W + PAD * 2 - FW) / S + 1) }, { FH * FW }>
    where T: Default + Clone + Copy + Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue {
    let image = image.as_raw_slice();

    let ys = (H + PAD * 2 - FH) / S + 1;
    let xs = (W + PAD * 2 - FW) / S + 1;

    let yskiped = PAD / FH;
    let xskiped = PAD / FW;
    let distance = FW + S - 1;
    let dist = (xs - xskiped) / distance * distance;

    let mut r = vec![T::default(); ys * FH * xs * FW];

    for y in yskiped..(ys - yskiped) {
        let pad = (PAD as isize - (y * FH) as isize).max(0) as usize;

        for ly in (0..FH).skip(pad).take((FH as isize + ((H + PAD) as isize - (FH + y * S) as isize).min(0)) as usize) {
            for x in (0..distance).skip(xskiped) {
                let dp = (PAD as isize - (x * distance) as isize).max(0) as usize * (x == 0) as usize;
                let sx = (PAD as isize - (x * S) as isize).max(0) as usize * (x == 0) as usize;

                let dst_start_offset = (y * FH * FW * xs + (ly * FW + x * FH * distance)) + dp;
                let src_start_offset = ((y * S + ly) - PAD) * W + x * S + sx - PAD;

                for (d,s) in (&mut r[
                    dst_start_offset..(dst_start_offset + FW - dp)
                ]).iter_mut().zip((&image[src_start_offset..(src_start_offset + FW)]).iter()) {
                    *d = *s;
                }

                let dst_start_offset = (y * FH * xs + ly + distance * FH) * FW + x * distance * FH;
                let src_start_offset = ((y * S + ly) - PAD) * W + x * S + distance - PAD;

                let dst_end_osffset = (y * FH * xs + ly + (xs - 1) * FH) * FW / distance * distance;
                let src_end_offset = src_start_offset + W - (x * S + distance) + PAD;

                for (d,s) in (&mut r[
                    dst_start_offset..dst_end_osffset
                ]).chunks_mut(distance).step_by(FH * distance).zip((&image[
                    src_start_offset..src_end_offset
                ]).chunks(distance)) {
                    for (d,s) in d.iter_mut().zip(s.iter()).take(FW) {
                        *d = *s;
                    }
                }
            }

            let dst_start_offset = (y * FH * xs + ly + dist * FH) * FW;
            let src_start_offset = ((y * S + ly) - PAD) * W + dist * S - PAD;

            for (d,s) in (&mut r[
                dst_start_offset..(dst_start_offset + FW)
            ]).iter_mut().zip((&image[src_start_offset..(src_start_offset + FW - PAD)]).iter()) {
                *d = *s;
            }
        }
    }

    return r.try_into().unwrap();
}