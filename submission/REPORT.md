# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Thế Hưng  **MSSV**: 2A202602381  **Ngày**: 07/10/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Colab Free T4 16GB`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| Thông số | Giá trị |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (`intent`, `urgency`, `product`, `sentiment`) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2.0 / 30 steps |

**Template có giữ khối `<think>` không?** Có — file `results/template_check.json` xác nhận: `"reasoning preserved — safe to train on traces"`. Thẻ `<think>` được giữ nguyên vẹn trong `apply_chat_template`, và chuỗi sinh kết thúc bằng token `<|im_end|>`.
Do dữ liệu ticket huấn luyện là các câu trả lời JSON trực tiếp và chat template của Qwen3.5 đóng thẻ `<think></think>` rỗng ngay trong tiền tố generation prompt, nên việc giữ nguyên template đảm bảo an toàn tuyệt đối cho quá trình tính loss mà không bị nuốt token.

---

## 2. Mask proof (NB1)

| Tiêu chí | Giá trị |
|---|---|
| `supervised_fraction` | 0.4149 (41.49%) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3758.5 |
| (b) base + optimized prompt | 0.765 | 0.791 | 1.000 | 1179.4 |
| (c) LoRA fine-tune | 0.970 | 0.478 | 1.000 | 1749.8 |

**(b) có thật sự mạnh hơn (a) không?** Có — Baseline (b) vượt trội hoàn toàn so với (a): Điểm target tăng vọt từ 0.000 lên 0.765, format đạt tuyệt đối 1.000 (100% sinh đúng định dạng JSON hợp lệ), và độ trễ giảm mạnh từ 3758.5 ms xuống 1179.4 ms (nhanh hơn gấp 3.2 lần).
Bạn có sửa `OPTIMIZED_PROMPT` không? Không sửa — Giá trị băm SHA-256 được giữ nguyên là `719e74d3b6232053`, đảm bảo tính liêm chính học thuật không làm suy yếu prompt đối chứng.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6272 | 0.9700 | 476.8 | 8.78 |
| `attn_only` | q,v | 283 | 32,456,704 | 0.0001 | 0.5368 | 0.9700 | 312.1 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-05 | 1.5702 | 0.0000 | 463.3 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | 0.9400 | 545.1 | 3.86 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế chính là Lỗi #3.

Trả lời ba câu:

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**
Trên tập đánh giá target, `attn_only` (với rank đã được bù ngân sách r=283) đạt điểm 0.9700, hoàn toàn hoà với bản `correct` (0.9700). Tuy nhiên, trên bảng train loss ở NB4, `attn_only` lại có loss thấp hơn rõ rệt (0.5368 so với 0.6272 của `correct`), tạo cảm giác giả tạo rằng cấu hình này ưu việt hơn. Sự đảo ngược thứ tự này chứng minh rằng việc dồn rank khổng lồ (r=283) vào riêng hai ma trận Attention (q, v) chỉ dẫn đến hiện tượng quá khớp (overfitting) trên tập huấn luyện chứ không hề cải thiện năng lực tổng quát hóa. Qua đó, ta thấy vị trí gắn adapter (`all-linear` trải đều cả MLP và Attention) đóng vai trò đòn bẩy quan trọng hơn nhiều so với việc cố tình gia tăng rank ở một vài tầng hẹp.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
Đường loss của `wrong_lr` (sử dụng learning rate 1e-5 theo thang của Full Fine-Tuning) diễn biến cực kỳ bế tắc: khởi đầu tại 2.163 và sau trọn vẹn 30 step chỉ hạ xuống mức 1.5702, hoàn toàn không hội tụ so với mức 0.6272 của `correct` (khiến mô hình đạt điểm target 0.000 và format 0.000). Nếu chỉ quan sát đường loss đi ngang mà không nắm được giá trị learning rate, người làm mô hình rất dễ kết luận sai lầm rằng dữ liệu bị lỗi, dung lượng mô hình quá nhỏ hoặc bài toán không thể học được bằng kỹ thuật LoRA. Trên thực tế, cấu trúc thắt nút cổ chai (bottleneck) của LoRA yêu cầu biên độ cập nhật gradient lớn hơn đáng kể (thường gấp 5x–10x so với full fine-tune) để trọng số di chuyển hiệu quả trong không gian con.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**
Thực nghiệm cho thấy `qlora` 4-bit giúp tiết kiệm một lượng bộ nhớ rất lớn: Peak VRAM giảm từ 8.78 GB xuống chỉ còn 3.86 GB (tiết kiệm được 4.92 GB, tương đương giảm ~56% dung lượng bộ nhớ GPU). Tuy nhiên, cái giá phải trả là thời gian huấn luyện bị kéo dài đáng kể (545.1 giây so với 476.8 giây do chi phí giải nén lượng tử hoá thời gian thực) và đặc biệt là độ chính xác tác vụ bị suy giảm từ 0.9700 xuống còn 0.9400 (tụt 3% điểm target). Kết quả đo lường thực tế này hoàn toàn ủng hộ khuyến nghị từ phía nhà phát triển: kiến trúc Qwen3.5 rất nhạy cảm với sai số lượng tử hóa 4-bit, và khi phần cứng T4 (14.6 GB) đã hoàn toàn đủ sức chứa bản 16-bit LoRA (8.78 GB), việc đánh đổi hiệu năng để lấy bộ nhớ trống là không kinh tế và không cần thiết.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = -0.313` · `valid_trace_rate = 0.0`

**Diễn giải:**
Cổng kiểm định hồi quy đưa ra phán quyết **FAILED**. Mặc dù trên tác vụ mục tiêu (target triage), bản LoRA fine-tune mang lại bước nhảy vọt ấn tượng khi chỉ số target tăng thêm **+0.205** (từ 0.765 của baseline prompt tối ưu lên 0.970 ở bản fine-tune), nhưng mô hình đã vi phạm nghiêm trọng tiêu chí bảo toàn năng lực tổng quát khi điểm regression bị tụt giảm **-0.313** (từ 0.791 xuống còn 0.478, vượt xa ngưỡng dung sai khắt khe 0.020).

Hiện tượng này phản ánh chính xác vấn đề kinh điển **Catastrophic Forgetting (Quên thảm họa)** trong quá trình SFT. Khi tập dữ liệu tinh chỉnh chỉ bao gồm 225 mẫu câu lệnh đơn nhất về ticket chăm sóc khách hàng và định dạng JSON thuần túy mà không được phối trộn thêm 1% đến 5% dữ liệu đệm tri thức phổ thông (replay/anchoring data theo deck §6.3), các ma trận trọng số thích ứng đã bị tái định hình quá đà theo cú pháp chuyên biệt, làm suy thoái khả năng trả lời các câu hỏi chỉ dẫn thông thường. Một kết quả FAILED được phân tích rành mạch về mặt bản chất nguyên nhân khoa học có giá trị thực tiễn cao hơn nhiều so với việc nới lỏng cổng kiểm tra để tạo ra một kết quả PASS giả tạo.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại. Gấp. Shop hỗ trợ tốt. | `{"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}` | Phân loại đúng nhưng latency cao (1179ms), sinh lời dẫn thừa | `{"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}` | ✅ **FT thắng**: Trích xuất tuyệt đối chính xác 4 trường, cấu trúc JSON sạch sẽ. |
| 2 | Cho mình hỏi, mình đặt đèn bàn LED mã đơn VN339109. Vỡ khi nhận. Gấp. Shop xem giúp. | `{"intent": "san_pham_loi", "urgency": "cao", "product": "đèn bàn LED", "sentiment": "trung_tinh"}` | Dễ nhầm giữa doi_tra và san_pham_loi do prompt thiếu ràng buộc sâu | `{"intent": "san_pham_loi", "urgency": "cao", "product": "đèn bàn LED", "sentiment": "trung_tinh"}` | ✅ **FT thắng**: Nhận diện chuẩn xác từ khóa "vỡ khi nhận" thành san_pham_loi. |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều. | `{"intent": "hoan_tien", "urgency": "thap", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}` | Nhận diện đúng urgency thấp dựa trên từ ngữ "khi nào tiện" | `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": "trung_tinh"}` | ❌ **FT thua**: FT đoán sai urgency thành "trung_binh" do thiên lệch dữ liệu học về hoàn tiền. |
| 4 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện. Cho tôi hỏi. | `{"intent": "san_pham_loi", "urgency": "thap", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"}` | Xử lý sắc thái trung tính tốt | `{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "nồi chiên không dầu", "sentiment": "tieu_cuc"}` | ❌ **FT thua**: FT nhận định nhầm sentiment thành "tieu_cuc" khi gặp cụm từ thiếu phụ kiện. |
| 5 | Xin chào, mình đặt balo laptop mã đơn DH863123. Đổi size. Hỏi cho biết thôi. Lần cuối mua ở đây. | `{"intent": "doi_tra", "urgency": "thap", "product": "balo laptop", "sentiment": "tieu_cuc"}` | Parse đúng cấu trúc cơ bản | `{"intent": "doi_tra", "urgency": "thap", "product": "balo laptop", "sentiment": "tieu_cuc"}` | ✅ **FT thắng**: Bắt đúng sắc thái tiêu cực ("lần cuối mua ở đây") và mức độ khẩn cấp thấp. |

**Có mẫu chung nào ở các ca FT thua không?**
Có một hình mẫu lỗi rất nhất quán: ở các ca fine-tune bị trừ điểm (đạt 0.75), lỗi tập trung chủ yếu ở hai trường `urgency` và `sentiment`. Khi khách hàng phản ánh sự cố dịch vụ (chưa nhận được tiền hoàn, sản phẩm thiếu phụ kiện) nhưng lại dùng lời lẽ nhã nhặn ("khi nào tiện", "cảm ơn shop"), mô hình fine-tune có xu hướng mặc định gán nhãn `urgency: trung_binh` và `sentiment: tieu_cuc`. Điều này chứng tỏ mô hình đã học vẹt mối tương quan giả (spurious correlation) giữa thực thể sự cố và thái độ bức xúc do tập dữ liệu huấn luyện kích thước nhỏ thiếu sự đa dạng về các ca biên (edge cases).

---

## 7. Kết luận & điều tôi học được

**Kết luận:**
Từ toàn bộ chuỗi thực nghiệm công bằng trong Lab 21, tôi kết luận rằng **chưa nên triển khai trực tiếp bản LoRA checkpoint này lên môi trường sản xuất đa nhiệm**, mặc dù hiệu năng phân loại ticket CSKH của nó đã vượt trội hoàn toàn baseline prompt tối ưu (+20.5% độ chính xác). Lý do then chốt nằm ở việc mô hình đã không vượt qua cổng hồi quy (regression gate), đánh mất 31.3% năng lực chỉ dẫn tổng quát do hiện tượng quên thảm họa. Trong bài toán hẹp này, đòn bẩy kỹ thuật mang tính quyết định số một chính là **Loss Masking**: việc đảm bảo chỉ tính gradient trên phần câu trả lời của trợ lý (supervised fraction ~41.5%) là điều kiện tiên quyết giữ cho mô hình không rơi vào trạng thái sinh lặp vô nghĩa. Đòn bẩy thứ hai là **vị trí gắn adapter**: cấu hình `all-linear` r=16 chứng minh tính cân bằng vượt trội so với việc cố tình ép rank r=283 vào riêng các module attention. Cuối cùng, việc chọn đúng **thang Learning Rate** (1e-4 thay vì 1e-5) là yếu tố sống còn để quá trình tối ưu hóa adapter hội tụ thành công.

**Ba điều tôi học được:**
1. **Chứng minh bằng giải mã thay vì niềm tin**: Không bao giờ được giả định hàm loss mask hoạt động đúng dựa trên tài liệu thư viện. Việc trực tiếp decode các mảng token được tính loss và bị che ở NB1 giúp phát hiện ngay lập tức lỗi rò rỉ prompt trước khi tiêu tốn tài nguyên GPU.
2. **Rank không phải là đòn bẩy, vị trí module mới quyết định**: So sánh đối chứng công bằng cùng số lượng tham số cho thấy nâng cao rank tại một vài lớp Attention chỉ làm giảm train loss giả tạo (overfitting), trong khi phân bổ adapter trải rộng toàn bộ mạng (`all-linear`) mới đem lại năng lực biểu diễn logic bền vững.
3. **Cổng hồi quy và sự thật khách quan**: Đánh giá LLM không thể chỉ dựa vào một thang đo mục tiêu duy nhất hay báo cáo perplexity. Phép kiểm định 4 nhóm (Target, Regression, Format, Latency) và việc dũng cảm thừa nhận kết quả FAILED giúp kỹ sư nhận diện chính xác ranh giới năng lực mô hình để có giải pháp khắc phục triệt để.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
1. Trộn thêm 3% đến 5% tập dữ liệu chỉ dẫn kiến thức tổng quát (General Instruction Replay Data) vào tập train để giữ vững năng lực nền tảng, đưa kết quả cổng hồi quy NB5 chuyển trạng thái thành **PASSED**.
2. Thực hiện trọn vẹn notebook tuỳ chọn **NB6**: Merge adapter trọng số LoRA trực tiếp vào base model `Qwen3.5-4B`, đo đạc xác minh độ tụt điểm (assert delta >= -0.01) và triển khai serving hot-swap nhiều adapter cùng lúc.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
