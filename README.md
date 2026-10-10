# Hệ thống chẩn đoán bệnh trên lá sầu riêng bằng AI

Notebook nghiên cứu thử nghiệm quy trình chẩn đoán bệnh trên lá sầu riêng bằng thị giác máy tính và mô hình ngôn ngữ lớn. Phiên bản hiện tại là **Cấu hình 1 (C1) — không sử dụng RAG**: mô hình thị giác phân tích ảnh, sau đó mô hình ngôn ngữ tạo chẩn đoán và tư vấn dựa trên kết quả đó cùng câu hỏi đầu vào.

> **Lưu ý:** Đây là dự án nghiên cứu/thử nghiệm, không phải công cụ chẩn đoán nông nghiệp đã được xác nhận để sử dụng độc lập. Nội dung tư vấn, đặc biệt tên thuốc, hoạt chất và liều lượng, cần được đối chiếu với tài liệu bảo vệ thực vật đáng tin cậy hoặc chuyên gia địa phương trước khi áp dụng.

## Mục tiêu

- Nhận diện một trong sáu lớp tình trạng/bệnh của lá sầu riêng.
- Sinh mô tả đặc điểm vết bệnh ở dạng JSON.
- Dùng mô hình ngôn ngữ tạo dòng chẩn đoán cuối cùng và phần giải thích/tư vấn bằng tiếng Việt.
- Đánh giá hiệu năng nhận diện, xem LLM có cải thiện hay làm sai nhãn VLM, và thống kê một số chỉ báo rủi ro trong phần tư vấn.

## Kiến trúc

```text
Ảnh lá sầu riêng
      │
      ├── Qwen2.5-VL-3B-Instruct + LoRA adapter
      │       ├── Nhãn bệnh
      │       └── Mô tả JSON: lesion_color, lesion_shape,
      │           distribution, severity
      │
      └── Nhãn + mô tả JSON + câu hỏi
                  │
                  ▼
          Qwen2.5-7B-Instruct
                  │
                  ▼
      Chẩn đoán cuối + tư vấn tiếng Việt
                  │
                  ▼
      Đánh giá và xuất kết quả
```

Notebook này chủ yếu thực hiện **suy luận và đánh giá** với mô hình Qwen2.5-VL đã fine-tune bằng LoRA từ trước; notebook không chứa quy trình huấn luyện adapter hoàn chỉnh. LLM không được cung cấp tài liệu tham chiếu trong C1, nên cấu hình này không phải RAG.

## Mô hình và công nghệ

- **VLM cơ sở:** `Qwen/Qwen2.5-VL-3B-Instruct`
- **VLM adapter:** checkpoint LoRA `checkpoint-1023` được nạp từ một Kaggle Input riêng.
- **LLM:** `Qwen/Qwen2.5-7B-Instruct`
- **Lượng tử hóa LLM:** 4-bit NF4 thông qua `bitsandbytes`.
- **Thư viện chính:** PyTorch, Transformers, PEFT, `qwen-vl-utils`, Accelerate, BitsAndBytes, NumPy, Pandas, Matplotlib, Pillow và tqdm.
- **Giao diện thử nghiệm:** Gradio (được mở trong chế độ chạy tương tác nếu môi trường cho phép).

## Sáu lớp được dự đoán

| Nhãn nguồn | Tên hiển thị trong notebook |
|---|---|
| `Algal` | Bệnh đốm rong (*Cephaleuros virescens*) |
| `Blight` | Bệnh cháy lá, chưa xác định tác nhân (`Leaf_Blight`) |
| `Colletotrichum` | Bệnh thán thư (*Colletotrichum* sp.) |
| `Phomopsis` | Bệnh đốm lá (*Phomopsis* sp.) |
| `Rhizoctonia` | Bệnh cháy lá, chết ngọn (*Rhizoctonia* sp.) |
| `Healthy` | Lá khỏe mạnh |

Tên lớp phản ánh ánh xạ nhãn được cài đặt trong notebook và dữ liệu đang dùng; không nên xem tên bệnh là kết luận xác nhận tác nhân gây bệnh trong mọi trường hợp.

## Môi trường chạy

Notebook hiện được viết cho **Kaggle Notebook có GPU NVIDIA**. Code có `assert torch.cuda.is_available()` và dùng các đường dẫn `/kaggle/input/...`, do đó chưa thể chạy nguyên trạng trên máy Windows/CPU hoặc môi trường khác nếu chưa sửa đường dẫn và cấu hình thiết bị.

Kết quả lịch sử trong notebook được tạo trên GPU Tesla T4 với khoảng 14,6 GiB VRAM. Nhu cầu bộ nhớ có thể khác theo phiên bản thư viện, cấu hình batch, kích thước ảnh và mô hình thực tế.

### Dữ liệu và model cần gắn vào Kaggle

Hãy tạo Kaggle Notebook, bật GPU trong phần Settings, sau đó đính kèm các dataset/model mà notebook đang tham chiếu. Các đường dẫn đang khai báo ở phần cấu hình đầu notebook là:

```python
DATASET_ROOT = "/kaggle/input/datasets/hhaaad/a-adurian-leaf-diseases1/Durian_Leaf_Diseases"
TEST_JSONL_PATH = "/kaggle/input/datasets/hhaaad/testjson05/test.jsonl"
VLM_ADAPTER_PATH = "/kaggle/input/notebooks/hauhoccode/vlm-train/qwen2.5-vl-durian-adapter/checkpoint-1023"
OUT_DIR = "/kaggle/working/config1"
```

Các đường dẫn này phụ thuộc vào tài nguyên được gắn vào Kaggle. Nếu bạn dùng dataset hoặc adapter khác, hãy sửa đường dẫn cho đúng. `TEST_JSONL_PATH` được dùng để lấy trường `question` của task `disease`; nhãn thật đang được suy ra từ tên thư mục/tên ảnh theo logic trong notebook. Có thể gắn thêm dữ liệu kiểm thử ngoài tại biến `EXT_DIR` để chạy phần kiểm thử bổ sung.

**Quan trọng:** Cần có quyền truy cập adapter LoRA và dữ liệu tương ứng. Chỉ tải file notebook lên GitHub không đồng nghĩa với việc người khác có thể chạy lại mô hình, vì checkpoint và dataset không nằm trong notebook này.

## Cách chạy

1. Mở notebook `test10v1.ipynb` trong Kaggle.
2. Bật GPU và đính kèm các dataset/model cần thiết như phần trên.
3. Kiểm tra các biến đường dẫn ở mục cấu hình đầu notebook. Đảm bảo `DATASET_ROOT`, `TEST_JSONL_PATH` và `VLM_ADAPTER_PATH` tồn tại trong phiên.
4. Kiểm tra `EVAL_LIMIT`, `ADVICE_N`, `RESUME` và các tham số batch. Để thử nhanh, đặt `EVAL_LIMIT` nhỏ và `ADVICE_N` nhỏ trước khi chạy toàn bộ.
5. Chạy các cell theo thứ tự từ trên xuống. Cell cài đặt sẽ gỡ một số gói và cài lại Transformers cùng các thư viện liên quan; nên thực hiện trong một notebook/session dành riêng cho dự án.
6. Kiểm tra các chỉ số và tệp đầu ra trong `/kaggle/working/config1/`.
7. Nếu muốn dùng phần Gradio, chạy trong chế độ tương tác. Code dùng `share=True`, vì vậy liên kết chia sẻ có thể công khai trong thời gian phiên còn chạy; không đưa ảnh hoặc dữ liệu nhạy cảm vào demo công khai.

Notebook tải model từ Hugging Face khi cần, vì vậy môi trường cần có quyền truy cập model hoặc các tệp model đã được lưu sẵn. Phiên Kaggle có thể bị giới hạn thời gian và bộ nhớ GPU.

## Tham số quan trọng

Các tham số có thể điều chỉnh ở phần cấu hình đầu notebook:

- `EVAL_LIMIT`: giới hạn số ảnh đánh giá; `None` nghĩa là chạy toàn bộ ảnh hợp lệ.
- `ADVICE_N`: số ảnh lấy mẫu phân tầng để sinh câu tư vấn đầy đủ và tính các chỉ số Mức 1; hiện đặt `120`, tương đương 20 ảnh mỗi lớp khi đủ dữ liệu.
- `VLM_BATCH`, `LLM_BATCH`: kích thước batch; giảm nếu gặp lỗi hết bộ nhớ GPU.
- `USE_TEST_QUESTIONS`: dùng câu hỏi từ `test.jsonl` làm đầu vào LLM hay dùng câu hỏi trung tính.
- `EXCLUDE_DUP_TEST`: loại ảnh test có nội dung trùng với ảnh trong các thư mục dữ liệu khác theo MD5 trong bước kiểm tra dự phòng.
- `REFERENCE_PATH`: đường dẫn `disease_reference.json`. Hiện đang là `None`; nếu chưa có tài liệu chuẩn thì Mức 2 không chạy.

## Kết quả lịch sử trong notebook

Các con số dưới đây được lấy từ lần chạy đã lưu trong chính notebook này; chúng là kết quả thực nghiệm hiện tại, **không phải bảo đảm hiệu năng trên dữ liệu mới**.

| Hạng mục | Kết quả đã lưu |
|---|---:|
| Ảnh test ban đầu | 402 |
| Ảnh đưa vào đánh giá sau khi loại trùng MD5 ở bước dự phòng | 398 |
| VLM — accuracy nhãn từ ảnh | 89,45% (356/398) |
| VLM — Macro-F1 | 0,893 |
| LLM — accuracy dòng chẩn đoán cuối | 86,43% (344/398) |
| LLM — Macro-F1 | 0,857 |
| So sánh cặp VLM/LLM (McNemar exact) | 21 trường hợp làm sai nhãn đúng, 9 trường hợp sửa đúng; p = 0,0428 |
| Sinh tư vấn đầy đủ để tính chỉ số Mức 1 | 120 ảnh |
| Độ chính xác chẩn đoán trên tập 120 ảnh tư vấn | 86,7% |

Trong lần chạy này, LLM làm giảm accuracy tổng thể so với nhãn VLM. Vì vậy, cần xem lại prompt/chính sách giữ nhãn sơ bộ và phân tích riêng các cặp lớp thường bị nhầm trước khi coi kết quả là tốt hơn.

## Giới hạn và việc cần làm trước khi báo cáo kết quả

1. **Có nguy cơ rò rỉ nhãn qua câu hỏi.** Log ghi nhận 10/394 câu hỏi task `disease` có nêu rõ tên một bệnh; cả VLM và LLM đều đạt 100% trên nhóm 10 câu này trong bảng phân tích. Để đánh giá công bằng, nên loại các trường hợp đó khỏi so sánh cuối cùng hoặc viết lại câu hỏi không chứa đáp án, rồi chạy lại toàn bộ chỉ số.
2. **Chưa xác minh đầy đủ rò rỉ từ dữ liệu fine-tune.** `TRAIN_JSONL_PATH` và `VAL_JSONL_PATH` đang để `None`, vì vậy notebook chỉ thực hiện kiểm tra trùng ảnh dự phòng giữa các thư mục. Bước đó phát hiện và loại 4 ảnh test trùng nội dung với ảnh ở thư mục khác, nhưng không chứng minh rằng các ảnh còn lại chưa từng xuất hiện trong dữ liệu fine-tune của adapter. Cần kiểm tra manifest/train JSONL gốc để xác nhận.
3. **Chưa chấm ảo giác dựa trên tài liệu tham chiếu.** `REFERENCE_PATH = None` nên notebook chỉ chạy các chỉ số tự động như tỷ lệ câu có nhắc hoạt chất, liều lượng hoặc thời gian. Đây là chỉ báo rủi ro, không phải tỷ lệ ảo giác đã được kiểm chứng. Muốn chạy Mức 2, cần hoàn thiện `disease_reference.json` từ nguồn chuyên môn, cấu hình `REFERENCE_PATH` rồi chạy lại.
4. **LLM có thể đổi nhãn VLM thành nhãn sai.** Trong lần chạy hiện tại, 21 nhãn đúng của VLM bị đổi thành sai, trong khi chỉ có 9 nhãn sai được sửa đúng.
5. **Đường dẫn phụ thuộc Kaggle và checkpoint ngoài notebook.** Muốn người khác tái lập được, cần ghi rõ nguồn dữ liệu, cách lấy adapter, phiên bản thư viện và cấu hình GPU; không nên đưa checkpoint/dataset có bản quyền hoặc hạn chế chia sẻ lên repository nếu chưa có quyền.
6. **Tư vấn thuốc chưa được xác minh chuyên môn.** Không sử dụng trực tiếp tên hoạt chất/liều lượng do LLM tạo ra làm hướng dẫn xử lý cây trồng nếu chưa kiểm chứng với nguồn được phép áp dụng tại địa phương.

## Tệp đầu ra

Theo danh sách được notebook ghi lại, thư mục `/kaggle/working/config1/` có thể chứa:

- `eval_c1.csv`: kết quả dự đoán/nhãn thật trên tập đánh giá.
- `summary_c1.csv`: bảng tổng hợp chỉ số chính.
- `answers_c1_diag.jsonl`: câu trả lời ngắn dùng đánh giá dòng chẩn đoán của LLM.
- `answers_c1.jsonl`: câu trả lời tư vấn đầy đủ trên tập con.
- `vlm_c1_label.jsonl`: đầu ra nhãn thô và nhãn đã ánh xạ từ VLM.
- `vlm_c1_desc.jsonl`: mô tả vết bệnh do VLM sinh ra.
- `advice_level1_c1.csv`: các chỉ số tự động về câu tư vấn.
- `confusion_C1_chẩn_đoán_cuối_của_LLM.png`: biểu đồ ma trận nhầm lẫn.
- `disease_reference_TEMPLATE.json`: mẫu để nhập tri thức tham chiếu.
- `run_meta_c1.json`: thông tin cấu hình/môi trường của lần chạy.

Một số tệp chỉ xuất hiện khi cell tương ứng được chạy. Các kết quả trong `/kaggle/working/` cần được tải xuống hoặc lưu thành phiên bản Kaggle nếu muốn giữ lại sau khi môi trường kết thúc.

## Hướng phát triển

- Loại hoặc sửa các câu hỏi tiết lộ nhãn và đánh giá lại trên tập test sạch.
- Đối chiếu ảnh test với manifest dữ liệu fine-tune thực tế.
- Thêm `disease_reference.json` có trích dẫn nguồn chuyên môn để kiểm tra mệnh đề tư vấn.
- So sánh VLM trực tiếp với hệ thống VLM + LLM trên cùng tập ảnh sạch; báo cáo accuracy, Macro-F1, confusion matrix và kiểm định thống kê.
- Giảm xu hướng LLM tự đổi nhãn khi bằng chứng thị giác không mâu thuẫn rõ ràng; cân nhắc đầu ra “không chắc chắn” khi dữ liệu ảnh không đủ.
- Tách mã nguồn thành các module Python, bổ sung `requirements.txt`, kiểm thử tự động và hướng dẫn tái lập để dự án dễ bảo trì hơn.

## Tuyên bố sử dụng

Mã nguồn và kết quả trong repository phục vụ học tập/nghiên cứu. Chẩn đoán từ ảnh có thể sai do điều kiện ánh sáng, góc chụp, chất lượng ảnh, bệnh đồng thời hoặc khác biệt ngoài tập dữ liệu. Kết quả không thay thế việc kiểm tra trực tiếp cây và ý kiến của cán bộ chuyên môn.
