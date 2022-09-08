use std::process::Output;

#[derive(Debug, Default)]
pub struct Lpc<T: num_traits::Float> {
    // 次数
    order: usize,
    // 自己相関関数
    acf: Vec<T>,
    // 線形予測係数
    coef: Vec<T>,

    // 最小誤差
    e: Vec<T>,

    // 線形予測係数を一時的に格納する
    temp_coef: Vec<T>,
}
impl<T: num_traits::Float> Lpc<T> {
    pub fn new(order: usize) -> Self {
        Self {
            order,
            acf: Vec::with_capacity(order + 1),
            coef: Vec::with_capacity(order + 1),
            e: Vec::with_capacity(order + 1),
            temp_coef: Vec::with_capacity(order + 1),
        }
    }
    pub fn order(&mut self, order: usize) {
        self.order = order;
        self.acf.clear();
        self.coef.clear();
        self.e.clear();
        self.temp_coef.clear();
    }
    pub fn calc(&mut self, data: &[T]) {
        auto_corr(data, self.order, &mut self.acf);
        levinson_durbin(
            &self.acf,
            self.order,
            &mut self.coef,
            &mut self.e,
            &mut self.temp_coef,
        );
    }
    pub fn coef(&self) -> &[T] {
        &self.coef
    }
    /// 線形予測係数から予測誤差をもとめる
    /// https://en.wikipedia.org/w/index.php?title=Linear_prediction&oldid=1099587626#The_prediction_model
    pub fn prediction_error(&self, input: &[T], pred_err: &mut [T]) {
        assert_eq!(input.len(), pred_err.len());
        let len = input.len();
        for (t, y) in pred_err.iter_mut().enumerate() {
            *y = self
                .coef
                .iter()
                .zip(input.iter().rev().skip(len - t)) // zipする
                .map(|(coef, input)| *input * *coef)
                .fold(T::zero(), |x, y| x + y);
        }
    }
    pub fn inverse_prediction_error(&self, pred_err: &[T], output: &mut [T]) {
        assert_eq!(output.len(), pred_err.len());
        let len = output.len();
        for (o, p) in output.iter_mut().zip(pred_err.iter()) {
            *o = *p;
        }
        for t in 0..len {
            output[t] = output[t]
                - self
                    .coef
                    .iter()
                    .skip(1)
                    .zip(output.iter().rev().cycle().skip(len - t)) // zipする
                    .map(|(coef, input)| *input * *coef)
                    .fold(T::zero(), |x, y| x + y);
        }
    }
}
fn auto_corr<T>(data: &[T], order: usize, acf_buf: &mut Vec<T>)
where
    T: num_traits::Float,
{
    // 自己相関関数
    acf_buf.clear();
    for i in 0..order + 1 {
        acf_buf.push(
            data.iter()
                .zip(data.iter().cycle().skip(i))
                .map(|(x, y)| x.mul(*y))
                .fold(T::zero(), |x, y| x + y),
        );
    }
}
fn levinson_durbin<T: num_traits::Float>(
    acf: &[T],
    order: usize,
    coef: &mut Vec<T>,

    e: &mut Vec<T>,
    temp_coef: &mut Vec<T>,
) {
    // Levinson-Durbinのアルゴリズム
    // k次のLPC係数からk+1次のLPC係数を再帰的に計算して

    // LPC係数を求める
    // LPC係数（再帰的に更新される）
    // a[0]は1で固定のためlpcOrder個の係数を得るためには+1が必要
    coef.clear();
    coef.push(T::one());
    coef.push(-acf[1] / acf[0]);

    // 最小誤差
    e.clear();
    e.push(T::one());
    e.push(acf[0] + acf[1] * coef[1]);

    // kの場合からk=1の場合までを再帰的に求める
    for k in 1..order {
        //lamdaを更新
        let mut lambda = T::zero();
        for j in 0..=k {
            lambda = lambda - coef[j] * acf[k + 1 - j];
        }
        lambda = lambda / e[k];
        // aを更新
        // UとVからaを更新
        temp_coef.clear();
        for c in coef.iter() {
            temp_coef.push(*c);
        }

        coef.clear();
        coef.push(T::one());
        for (u, v) in temp_coef[1..=k]
            .iter()
            .zip(temp_coef.iter().skip(1).take(k).rev())
        {
            coef.push(*u + lambda * *v)
        }
        coef.push(lambda);

        e.push(e[k] * (T::one() - lambda * lambda));
    }
}
// 周波数応答
// fn freqz(b: &[f32], a: &[f32], buf: &mut [f32]) {
//     let n2pi = 2.0 * PI / buf.len() as f32;
//     for (i, h) in buf.iter_mut().enumerate() {
//         let z = (1.0, (i as f32 * -n2pi)).exp();
//         let b_len = b.len();
//         let numerator: Complex<_> = b
//             .iter()
//             .enumerate()
//             .map(|(i, x)| x * z.powi((b_len - i) as i32))
//             .sum();
//         let a_len = a.len();
//         let denominator: Complex<_> = a
//             .iter()
//             .enumerate()
//             .map(|(i, x)| x * z.powi((a_len - i) as i32))
//             .sum();

//         *h = (numerator / denominator).abs();
//     }
// }
