use realfft::num_complex::{Complex, ComplexFloat};
use std::f32::consts::PI;

#[derive(Debug, Default)]
pub struct Lpc {
    // 次数
    order: usize,
    // 自己相関関数
    acf: Vec<f32>,
    // 線形予測係数
    coef: Vec<f32>,

    e: Vec<f32>,
    u: Vec<f32>,
    v: Vec<f32>,
}
impl Lpc {
    pub fn new(order: usize) -> Self {
        Self {
            order,
            acf: Vec::with_capacity(order + 1),
            coef: Vec::with_capacity(order + 1),
            e: Vec::with_capacity(order + 1),
            u: Vec::with_capacity(order + 2),
            v: Vec::with_capacity(order + 2),
        }
    }
    pub fn order(&mut self, order: usize) {
        self.order = order;
        self.acf.clear();
        self.coef.clear();
        self.e.clear();
        self.u.clear();
        self.v.clear();
    }
    pub fn calc(&mut self, data: &[f32]) {
        self.auto_corr(data);
        self.levinson_durbin();
    }
    pub fn coef(&self) -> &[f32] {
        &self.coef
    }
    pub fn prediction_error(&self, input: &[f32], pred_err: &mut [f32]) {
        assert_eq!(input.len(), pred_err.len());
        let len = input.len();
        for (t, y) in pred_err.iter_mut().enumerate() {
            *y = self
                .coef
                .iter()
                .enumerate()
                .skip(1)
                // TODO: インデクスアクセスをなくせば最適化できるかも
                .map(|(i, coef)| input[(len + t - i) % len] * coef)
                .sum();
        }
    }
    fn auto_corr(&mut self, data: &[f32]) {
        // 自己相関関数
        self.acf.clear();
        for i in 0..self.order + 1 {
            self.acf.push(
                data.iter()
                    .zip(data.iter().cycle().skip(i))
                    .map(|(x, y)| x * y)
                    .sum(),
            );
        }
    }
    fn levinson_durbin(&mut self) -> &[f32] {
        // Levinson-Durbinのアルゴリズム
        // k次のLPC係数からk+1次のLPC係数を再帰的に計算して

        // LPC係数を求める
        // LPC係数（再帰的に更新される）
        // a[0]は1で固定のためlpcOrder個の係数を得るためには+1が必要
        self.coef.clear();
        self.coef.push(1.0);
        self.coef.push(-self.acf[1] / self.acf[0]);

        // 最小誤差
        self.e.clear();
        self.e.push(1.0);
        self.e.push(self.acf[0] + self.acf[1] * self.coef[1]);

        // kの場合からk=1の場合までを再帰的に求める
        for k in 1..self.order {
            self.u.clear();
            self.v.clear();

            //lamdaを更新
            let mut lambda = 0.0;
            for j in 0..=k {
                lambda -= self.coef[j] * self.acf[k + 1 - j];
            }
            lambda /= self.e[k];
            // aを更新
            // UとVからaを更新
            self.u.push(1.0);
            self.v.push(0.0);
            for x in &self.coef[1..=k] {
                self.u.push(*x);
            }
            for x in self.coef.iter().skip(1).take(k).rev() {
                self.v.push(*x);
            }
            self.u.push(0.0);
            self.v.push(1.0);

            self.coef.clear();
            for (u, v) in self.u.iter().zip(&self.v) {
                self.coef.push(u + lambda * v)
            }

            self.e.push(self.e[k] * (1.0 - lambda * lambda));
        }
        &self.coef
    }
}
// 周波数応答
fn freqz(b: &[f32], a: &[f32], buf: &mut [f32]) {
    let n2pi = 2.0 * PI / buf.len() as f32;
    for (i, h) in buf.iter_mut().enumerate() {
        let z = Complex::new(0.0, i as f32 * -n2pi).exp();
        let b_len = b.len();
        let numerator: Complex<_> = b
            .iter()
            .enumerate()
            .map(|(i, x)| x * z.powi((b_len - i) as i32))
            .sum();
        let a_len = a.len();
        let denominator: Complex<_> = a
            .iter()
            .enumerate()
            .map(|(i, x)| x * z.powi((a_len - i) as i32))
            .sum();

        *h = (numerator / denominator).abs();
    }
}
