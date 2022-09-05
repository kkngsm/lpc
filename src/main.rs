use std::{f32::consts::PI, fs::File};

use realfft::num_complex::{Complex, ComplexFloat};

macro_rules! plot_with_name {
    ($title:expr, $($name_ys:expr), +) => {
        use plotly::common::{Mode, Title};
        use plotly::layout::Layout;
        use plotly::{Plot, Scatter};
        let layout = Layout::new().title(Title::new($title));
        let mut plot = Plot::new();
        $(
            let (name, ys) = $name_ys;
            let xs = (0..ys.len()).map(|i| i as f32);
            let trace = Scatter::new(xs, ys.to_owned())
                .mode(Mode::Lines)
                .name(name);
            plot.add_trace(trace);
        )+
        plot.set_layout(layout);
        plot.show();
    };
}
macro_rules! plot {
    ($title:expr, $($ys:expr), +) => {
        plot_with_name!($title, $((stringify!($ys), $ys)   ), +);
    }
}
fn main() {
    const LPC_ORDER: usize = 32;

    let original: Vec<f32> = serde_json::from_str(include_str!("./wave.json")).unwrap();
    let N = original.len();
    let mut planner = realfft::RealFftPlanner::new();
    let fft = planner.plan_fft_forward(N);
    let ifft = planner.plan_fft_inverse(N);

    let logspec: Vec<_> = {
        let mut spectrum = fft.make_output_vec();
        let mut ori = original.clone();
        fft.process(&mut ori, &mut spectrum);

        spectrum
            .into_iter()
            .map(|c| c.abs().log10() * 20.0)
            .collect()
    };

    let loglpcspec: Vec<_> = {
        let mut lpc = Lpc::new(LPC_ORDER);
        let lpcspec = lpc.calc(LPC_ORDER, &original);
        lpcspec
            .into_iter()
            .map(|c| c.abs().log10() * 20.0)
            .take(N / 2)
            .collect()
    };
    let cepstrum: Vec<_> = {
        let mut ori = original.clone();
        let mut spectrum = fft.make_output_vec();
        // cepstrum
        fft.process(&mut ori, &mut spectrum);
        for s in &mut spectrum {
            s.re = s.abs().log10() * 20.0;
            s.im = 0.0;
        }
        ifft.process(&mut spectrum, &mut ori);
        for x in &mut ori {
            *x /= N as f32;
        }
        let mut cepstrum = ori;

        for x in cepstrum.iter_mut().take(N / 2).skip(LPC_ORDER) {
            *x = 0.0;
        }
        for x in cepstrum.iter_mut().rev().take(N / 2).skip(LPC_ORDER) {
            *x = 0.0;
        }
        fft.process(&mut cepstrum, &mut spectrum);
        spectrum.into_iter().map(|c| c.re).take(N / 2).collect()
    };

    plot!("title", logspec, loglpcspec, cepstrum);
}

#[derive(Debug, Default)]
struct Lpc {
    acf: Vec<f32>,
    coef: Vec<f32>,
    e: Vec<f32>,
    u: Vec<f32>,
    v: Vec<f32>,
}
impl Lpc {
    pub fn new(order: usize) -> Self {
        Self {
            acf: Vec::with_capacity(order + 1),
            coef: Vec::with_capacity(order + 1),
            e: Vec::with_capacity(order + 1),
            u: Vec::with_capacity(order + 2),
            v: Vec::with_capacity(order + 2),
        }
    }
    pub fn calc(&mut self, order: usize, input: &[f32]) -> Vec<Complex<f32>> {
        self.auto_corr(order, input);
        self.levinson_durbin(order);
        let e = self.e.last().unwrap().sqrt();
        freqz(&[e], &self.coef, input.len())
    }
    fn auto_corr(&mut self, order: usize, data: &[f32]) {
        // 自己相関関数
        self.acf.clear();
        for i in 0..order + 1 {
            self.acf
                .push(data.iter().zip(&data[i..]).map(|(x, y)| x * y).sum());
        }
    }
    fn levinson_durbin(&mut self, order: usize) -> &[f32] {
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
        for k in 1..order {
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
fn freqz(b: &[f32], a: &[f32], n: usize) -> Vec<Complex<f32>> {
    let mut H = vec![Complex::default(); n];
    let n2pi = 2.0 * PI / n as f32;
    for (i, h) in H.iter_mut().enumerate() {
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

        *h = numerator / denominator;
    }

    return H;
}
