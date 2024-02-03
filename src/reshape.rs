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

    let yskiped = (PAD as isize - FH as isize).max(0) as usize / S + (PAD / FH).min(1);
    let xskiped = (PAD as isize - FW as isize).max(0) as usize / S + (PAD / FW).min(1);
    let distance = (FW + S - 1) / S * S;
    let xc = distance / S;
    let xc = xc.min(W + PAD * 2 - (xc - 1) * S);

    let mut r = vec![T::default(); ys * FH * xs * FW];

    for y in yskiped..(ys - yskiped) {
        let p = (PAD as isize - (y * S) as isize).max(0) as usize;

        for ly in (0..FH).skip(p).take((FH as isize + ((H + PAD) as isize - (FH + y * S) as isize).min(0)) as usize) {
            for x in 0..xc {
                let x = x + xskiped;

                let rp = PAD.min(x * S);
                let sx = (PAD as isize - (x * S) as isize).max(0).min(FW as isize) as usize * (x * S <= PAD) as usize;
                let cw = (FW - sx).min(W - (x * S - rp));
                let dst_start_offset = (y * FH * xs + (ly + x * FH)) * FW + sx;
                let src_start_offset = ((y * S + ly) - PAD) * W + x * S - rp;

                for (d,s) in (&mut r[
                    dst_start_offset..(dst_start_offset + FW - sx)
                ]).iter_mut().zip((&image[src_start_offset..(src_start_offset + cw)]).iter()) {
                    *d = *s;
                }

                let rp = PAD.min(x  * S + distance);

                if distance / S + x >= xs || x * S + distance >= W + PAD {
                    continue;
                }

                let dst_start_offset = (y * FH * xs + ly + distance / S * FH) * FW + x * FW * FH;
                let src_start_offset = ((y * S + ly) - PAD) * W + x * S + distance - rp;

                let dst_end_offset = (y * FH * xs) * FW + xs * FH * FW;
                let src_end_offset = src_start_offset + W + rp - (x * S + distance);

                for (d,s) in (&mut r[
                    dst_start_offset..dst_end_offset
                ]).chunks_mut(FW).step_by(FH * distance / S).zip((&image[
                    src_start_offset..src_end_offset
                ]).chunks(distance)) {
                    for (d,s) in d.iter_mut().zip(s.iter()) {
                        *d = *s;
                    }
                }
            }
        }
    }

    return r.try_into().unwrap();
}
