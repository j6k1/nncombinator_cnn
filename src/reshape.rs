use itertools::Itertools;
use nncombinator::arr::Arr2;
use nncombinator::error::SizeMismatchError;
use nncombinator::mem::AsRawSlice;

use crate::collection::{ImageView};

pub fn im2col<'a,T,const H:usize,const W:usize,const FH:usize,const FW:usize,const PAD:usize,const S:usize>(image:&ImageView<'a,T,H,W>)
    -> Result<Arr2<T,{ ((H + PAD * 2 - FH) / S + 1) * ((W + PAD * 2 - FW) / S + 1) }, { FH * FW }>,SizeMismatchError>
    where T: Default + Clone + Copy + Send + Sync + 'static {
    let image = image.as_raw_slice();

    let ys = (H + PAD * 2 - FH) / S + 1;
    let xs = (W + PAD * 2 - FW) / S + 1;

    let remain_width = (PAD + S - 1) / S;
    let remain_height = (PAD + S - 1) / S;
    let yskiped = PAD / FH;
    let xskiped = PAD / FW;
    let distance = (FW + S - 1);
    let dist = xs / distance * distance;

    let dst_width = xs - remain_width * 2 * FW / S - FH * FW;
    let src_width = W + PAD * 2 - remain_width * 2 - FW;

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

                for (d,s) in (&mut r[
                    dst_start_offset..((y * FH * xs + ly + (xs - 1) * FH) * FW / distance * distance)
                ]).chunks_mut(FW).step_by(FH * distance).zip((&image[
                    src_start_offset..(src_start_offset + W - (x * S + distance) + PAD)
                ]).chunks(S * distance)) {
                    for (d,s) in d.iter_mut().zip(s.iter()) {
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

    return r.try_into();
    for sx in [0, xs - remain_width * FW / S] {
        for y in ((remain_width * FH / S)..(ys - remain_width * FH / S - FH)).step_by(FH) {
            for ly in 0..FH {
                for ox in (0..(remain_width * FW / S)).step_by(S).enumerate().map(|(i, _)| i) {
                    let dst_start_offset = (y + ly) * FW + sx * ys + ox * FH * FW;
                    let src_start_offset = ((y * S / FH + ly).max(remain_width) - remain_width) * W + ox * S;

                    let it = (
                        &mut r[(dst_start_offset)..(dst_start_offset + remain_width * FW / S)]
                    ).chunks_mut(FW).step_by(FW / S).zip((&image[
                        (src_start_offset)..(src_start_offset + remain_width)
                    ]).chunks(FW));

                    for (d, s) in it {
                        for (d, s) in d.iter_mut()
                            .skip((PAD as isize - ((y + ly) * S / FH) as isize).max(0) as usize * FW +
                                (PAD as isize - (ox * S) as isize).max(0) as usize
                            )
                            .zip(s.iter()
                                 //.take(FW - ((ox * S + i * FW + FW) as isize - (xs * S / FW - PAD) as isize).max(0) as usize)
                            ) {
                            *d = *s;
                        }
                    }
                }
            }
        }
    }

    for y in (remain_width..(ys - remain_width - FH)).step_by(FH) {
        for ly in 0..FH {
            for ox in (0..FW).step_by(S).enumerate().map(|(i, _)| i) {
                let dst_start_offset = xs * y + remain_width * FW / S + ly * FW + ox * FH * FW;
                let src_start_offset = (y - remain_width + (remain_width - PAD)) / FH + ly * W + ox * S;

                let it = (
                    &mut r[(dst_start_offset)..(dst_start_offset + dst_width)]
                ).chunks_mut(FW).step_by(FW / S).zip((&image[
                    (src_start_offset)..(src_start_offset + src_width)
                ]).chunks(FW));

                if (W - FW) / FW >= 8 {
                    let chunk_size = (W - FW) / FW / 8;

                    let chunks = it.chunks(chunk_size);
                    let mut chunks_iter = chunks.into_iter();

                    for _ in 0..8 {
                        let it = chunks_iter.next().unwrap();

                        for (d, s) in it {
                            for (d,s) in d.iter_mut().zip(s.iter()) {
                                *d = *s;
                            }
                        }
                    }

                    let it = (
                        &mut r[(dst_start_offset)..(dst_start_offset + dst_width)]
                    ).chunks_mut(FW).step_by(FW / S).zip((&image[
                        (src_start_offset)..(src_start_offset + src_width)
                    ]).chunks(FW)).skip((W - FW) / FW / 8 * 8);

                    for (d, s) in it {
                        for (d,s) in d.iter_mut().zip(s.iter()) {
                            *d = *s;
                        }
                    }
                } else {
                    let it = (
                        &mut r[(dst_start_offset)..(dst_start_offset + dst_width)]
                    ).chunks_mut(FW).step_by(FW / S).zip((&image[
                        (src_start_offset)..(src_start_offset + src_width)
                    ]).chunks(FW).skip((W - FW) / FW / 8 * 8));

                    for (d, s) in it {
                        for (d,s) in d.iter_mut().zip(s.iter()) {
                            *d = *s;
                        }
                    }
                }
            }
        }
    }

    r.try_into()
}