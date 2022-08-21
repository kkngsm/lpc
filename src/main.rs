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
    const LPC_ORDER: usize = 16;

    let original = (0..128)
        .map(|i| {
            let i = i as f32;
            (i * 0.01).sin()
                + 0.75 * (i * 0.03).sin()
                + 0.5 * (i * 0.05).sin()
                + 0.25 * (i * 0.11).sin()
        })
        .collect::<Vec<_>>();
    // LPCで前向き予測した信号を求める
    let mut predicted = original.clone();
    // 過去lpcOrder分から予測するので開始インデックスはlpcOrderから
    // それより前は予測せずにオリジナルの信号をコピーしている

    let mut lpc = Lpc::new(LPC_ORDER);
    let a = lpc.calc(LPC_ORDER, &original);

    for (i, p) in predicted.iter_mut().enumerate().skip(LPC_ORDER) {
        *p = -original
            .iter()
            .skip(i)
            .take(LPC_ORDER)
            .zip(a.iter().skip(1))
            .map(|(x, y)| x * y)
            .sum::<f32>();
    }
    plot!("title", original, predicted);
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
    pub fn calc(&mut self, order: usize, input: &[f32]) -> &[f32] {
        self.auto_corr(order, input);
        self.levinson_durbin(order)
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
