use itertools::Itertools;

use nncombinator::mem::AsRawSlice;
use crate::collection::{ImageView};

pub fn im2col<'a,T,const H:usize,const W:usize,const FH:usize,const FW:usize,const PAD:usize,const S:usize>(image:&ImageView<'a,T,H,W>)
    -> Vec<T> where T: Default + Clone + Copy + Send + Sync + 'static {
    let image = image.as_raw_slice();

    let ys = ((H + PAD * 2 - FH) / S + 1) * FH;
    let xs = ((W + PAD * 2 - FW) / S + 1) * FW;

    let remain_width = (PAD + S - 1) / S * S;
    let dst_width = xs - remain_width * 2 * S / FW - FH * FW;
    let src_width = W - remain_width * 2 - FW;

    let mut r = vec![T::default(); ys * xs];

    for sy in [0,ys - remain_width - FH] {
        for y in (sy..(sy + remain_width)).step_by(FH) {
            for ly in 0..FH {
                for ox in (0..FW).step_by(S).enumerate().map(|(i, _)| i) {
                    let dst_start_offset = xs * y + ly * FW + ox * FH * FW;
                    let src_start_offset = (y.max(remain_width) - remain_width) * S / FH + ly * W + ox * S;

                    let it = (
                        &mut r[(dst_start_offset)..(dst_start_offset + dst_width)]
                    ).chunks_mut(FW).step_by(FH / (FW / S)).zip((&image[
                        (src_start_offset)..(src_start_offset + src_width)
                    ]).chunks(FW));

                    for (i, (d, s)) in it.enumerate() {
                        for (d, s) in d.iter_mut()
                            .skip((PAD - PAD.min(y + ly)) * FW + (PAD - PAD.min(ox + i) * S))
                            .zip(s.iter()
                                .take(FH - (PAD - (PAD as isize)
                                .max(PAD as isize - ys as isize - (y as isize + ly as isize)) as usize) +
                                   FW - (PAD - (PAD as isize).max(
                                PAD as isize - xs as isize - (ox + i) as isize * S as isize + FW as isize
                                   ) as usize
                                ))
                            ) {
                            *d = *s;
                        }
                    }
                }
            }
        }
    }

    for sx in [0, xs - FW * S / FW] {
        for y in (remain_width..(ys - remain_width - FH)).step_by(FH) {
            for ly in 0..FH {
                for ox in (0..FW).step_by(S).enumerate().map(|(i, _)| i) {
                    let dst_start_offset = xs * y + sx + ly * FW + ox * FH * FW;
                    let src_start_offset = (y - remain_width + (remain_width - PAD) ) * S / FH + ly * W + sx * S / FW + ox * S;

                    let it = (
                        &mut r[(dst_start_offset)..(dst_start_offset + remain_width * FH * FW)]
                    ).chunks_mut(FW).step_by(FH / FW).zip((&image[
                        (src_start_offset)..(src_start_offset + remain_width)
                    ]).chunks(FW));

                    for (i, (d, s)) in it.enumerate() {
                        for (d, s) in d.iter_mut()
                            .skip((PAD - PAD.min(y + ly)) * FW + (PAD - PAD.min(ox + i) * S))
                            .zip(s.iter()
                                .take(FH - (PAD - (PAD as isize)
                                .max(PAD as isize - ys as isize - (y as isize + ly as isize)) as usize) +
                                    FW - (PAD - (PAD as isize).max(
                                    PAD as isize - xs as isize - (ox + i) as isize * S as isize + FW as isize
                                    ) as usize
                                ))
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
                let dst_start_offset = xs * y + remain_width * S / FW + ly * FW + ox * FW;
                let src_start_offset = y - remain_width * S / FH + ly * W + ox * S;

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
                    ).chunks_mut(FW).step_by(FW / S).skip((W - FW) / FW / 8 * 8).zip((&image[
                        (src_start_offset)..(src_start_offset + src_width)
                    ]).chunks(FW).skip((W - FW) / FW / 8 * 8));

                    for (d, s) in it {
                        for (d,s) in d.iter_mut().zip(s.iter()) {
                            *d = *s;
                        }
                    }
                } else {
                    let it = (
                        &mut r[(dst_start_offset)..(dst_start_offset + dst_width)]
                    ).chunks_mut(FW).step_by(FW / S).skip((W - FW) / FW / 8 * 8).zip((&image[
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

    r
}