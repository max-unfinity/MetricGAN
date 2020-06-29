# MetricGAN Research for Speech Enhancement

*(MetricGAN: Generative Adversarial Networks based Black-box Metric Scores
Optimization for Speech Enhancement. **Szu-Wei Fu et al.**)*

*Paper on arxiv: https://arxiv.org/pdf/1905.04874.pdf*

---

Repository guide:
- Summary.ipynb - Отчет
- src/utils.py - Функции для создания датасета, предобработки и работы с аудио
- src/models.py - Классы моделей и метрик
- src/MetricGAN train.ipynb - Основной ноутбук. Обучение и создание датасета

---

Requirements:
- [pystoi](https://github.com/mpariente/pystoi)
- [pesq](https://github.com/ludlows/python-pesq)
- [spectral_normalization.py](https://github.com/christiancosgrove/pytorch-spectral-normalization-gan/blob/master/spectral_normalization.py)
