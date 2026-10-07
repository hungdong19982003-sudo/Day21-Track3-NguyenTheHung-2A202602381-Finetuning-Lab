# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**
Sự tương phản gay gắt giữa train loss và task accuracy ở thí nghiệm `attn_only` (NB4): khi dồn rank cực lớn r=283 vào riêng 2 tầng attention (q,v) để khớp ngân sách tham số với bản `correct`, train loss của nó giảm sâu hơn hẳn (0.5368 so với 0.6272). Nhìn qua biểu đồ loss cứ ngỡ `attn_only` ưu việt hơn, nhưng khi đo trên tập target task ở NB5 thì điểm accuracy chỉ bằng nhau (0.9700). Điều này cho thấy dồn rank lớn vào tầng hẹp chỉ gây overfitting trên tập train. Đồng thời, việc mô hình bị quên thảm họa (regression score tụt -31.3%) chỉ sau 30 step huấn luyện trên tập dữ liệu hẹp 225 mẫu cũng là một bất ngờ lớn về rủi ro của SFT.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**
Tôi mất nhiều thời gian nhất ở công đoạn sinh văn bản (text generation / autoregressive decoding) khi chạy đánh giá ở NB2 và NB5 (tổng cộng mất gần 40–50 phút trên GPU T4) chứ không phải lúc huấn luyện LoRA ở NB3 (~8 phút). Trước khi thực hành, tôi đinh ninh quá trình huấn luyện backward pass sẽ chiếm 80% thời lượng. Nhưng vì lab thiết kế hệ thống 3-baseline chặt chẽ, tập eval phải được giải mã tham lam (greedy decode) 3 lần riêng biệt (baseline a, baseline b, và checkpoint LoRA) cộng thêm 4 run đối chứng ở NB4/NB5, khiến chi phí sinh văn bản trở thành nút thắt cổ chai lớn nhất.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**
Trước đây tôi tin vào hai điều:
- Một là: "Rank LoRA càng lớn thì adapter càng mạnh mẽ và thông minh".
- Hai là: "Chỉ cần train loss giảm đều đặn là mô hình đang học tốt và tiến bộ".
Sau lab này, tôi nhận ra rank thực chất là dung lượng biểu diễn (capacity) tương ứng với độ phức tạp thông tin của tập dữ liệu; đòn bẩy kiến trúc thực sự nằm ở vị trí cắm adapter (`all-linear` bao phủ cả Attention lẫn MLP). Train loss thấp hoàn toàn có thể là ảo giác học vẹt, và một quy trình đánh giá nghiêm túc bắt buộc phải có cổng hồi quy (regression gate) đa khía cạnh thay vì chỉ nhìn vào loss hay perplexity.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**
Tôi dùng AI assistant để giải thích chat template Jinja của tokenizer Qwen3.5, đối chiếu logic slice token tính loss mask ở NB1 và viết các script parse JSON, phân tích bảng số liệu.
Chỗ AI sai: Khi mô hình không hội tụ hoặc cần cải thiện kết quả, AI thường đưa ra lời khuyên theo lối mòn là "hãy nâng rank LoRA lên 64 hoặc 128" và "tăng thêm số epoch". Lời khuyên đó bỏ qua hoàn toàn nguyên tắc so sánh công bằng về ngân sách tham số và thực tế là `all-linear` r=16 đã giải quyết trọn vẹn bài toán. Ngoài ra, AI cũng có xu hướng đề xuất đánh giá chất lượng mô hình chỉ bằng loss/perplexity mà không hề cảnh báo về hiện tượng quên thảm họa năng lực tổng quát.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**
Bước đầu tiên chắc chắn không phải là chuẩn bị GPU để train ngay, mà là: **Xây dựng tập dữ liệu đánh giá chuẩn (Golden Evaluation Set) bao gồm cả tác vụ mục tiêu lẫn bài test hồi quy năng lực nền tảng, sau đó thiết lập Baseline (b) bằng Prompt Engineering tử tế**.
Tôi sẽ tối ưu hóa system prompt, few-shot và schema constraint trên base model nguyên bản trước để đo trần năng lực của prompting. Nếu prompt tối ưu đã giải quyết được 80–90% SLA của khách hàng với chi phí và độ trễ chấp nhận được, tôi sẽ tư vấn khách hàng dừng lại ở prompting. Chỉ khi prompting không thể đáp ứng được (yêu cầu định dạng JSON tuyệt đối 100%, giảm triệt để độ trễ và token prompt) thì mới bắt tay vào fine-tune — và khi đó, tôi đã có sẵn một mốc baseline vững chắc để chứng minh giá trị thực tế của giải pháp fine-tune.
