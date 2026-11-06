# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** GPU T4 (`Device: cuda`)

**Python / PyTorch / Ultralytics:** Python 3.13.16 / PyTorch 2.11.0+cu130 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt` (cả ba đều báo `PASS` khi kiểm tra checksum)

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không thay đổi code, checkpoint hoặc ngưỡng. Notebook chạy từ bản `v1.0.3`.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**File đã dùng:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`, với `sample_id = traffic`, `rank = 1`.

1. **Record hạng 1:** `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh**. Trường dữ liệu không có box hay polygon để chỉ ra một vật cụ thể, và `task` là `image_classification`. Ảnh có nhiều xe buýt, ô tô và người đi bộ, nhưng cả ảnh chỉ nhận một danh sách top-5 gồm `cab`, `minibus`, `police_van`, `recreational_vehicle`, `streetcar`.
3. Danh sách lớp **đi kèm taxonomy/checkpoint**, model không tự nghĩ ra. `yolo11n-cls.pt` chỉ có thể chọn trong các lớp ImageNet-1K.
4. Cùng một tên có thể có ID khác nhau ở taxonomy khác. Ví dụ trong output lab, `dining_table` là `class_id` 532 trong ImageNet-1K, còn `dining table` là `class_id` 60 trong COCO-80. Nếu chỉ ghi tên lớp, ta khó biết nhãn thuộc danh sách nào và dễ ghép nhầm dữ liệu.
5. Guideline cần nói rõ nhãn đang mô tả điều gì và cách chọn khi có nhiều vật. Ví dụ: chọn chủ đề chính của ảnh hoặc vật chiếm phần lớn diện tích. Nếu có hai chủ đề ngang nhau hoặc không có lớp phù hợp, annotator cần escalation thay vì tự chọn. Với ảnh `traffic`, guideline phải trả lời được nhãn nên là cảnh giao thông hay một loại xe, vì `cab` chỉ mô tả một loại xe trong khi ảnh có nhiều loại.
6. `score` chỉ là mức tin của model, không phải xác nhận của con người. Ở đây `cab` có score 0.510915 và đứng hạng 1, nhưng trên ảnh tôi thấy rõ nhiều xe buýt và ô tô hơn, chứ không thấy rõ taxi. Ground truth phải do người gán theo guideline rồi được kiểm tra lại.

## 2. Phát hiện vật thể

1. **Record đã chọn:** `sample_id`=kitchen ; `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Gốc tọa độ `(0, 0)` nằm ở góc trên-trái ảnh. Góc trên-trái của box ở `(385.33, 69.24)`, nghĩa là cách mép trái khoảng 385 px và cách mép trên khoảng 69 px. Góc dưới-phải ở `(498.92, 348.92)`. Box rộng khoảng 113.58 px, cao khoảng 279.68 px và bao quanh người đầu bếp đứng quay lưng ở bên phải ảnh.
3. **Số prediction theo ngưỡng:** ngưỡng `0.20` có **17** prediction, ngưỡng `0.60` có **6** prediction. Để đối chiếu, ngưỡng `0.35` có 11 prediction, đúng bằng số record `kitchen` trong JSON.
4. Khi hạ từ `0.60` xuống `0.20`, reviewer phải xem thêm 11 box. Với mỗi box, reviewer kiểm tra có đúng vật thể không, tên lớp có đúng không, box có sát không và có trùng với box khác không. Reviewer cũng phải đối chiếu với ảnh để tìm vật còn bị bỏ sót. Ví dụ `dining table` chỉ xuất hiện ở ngưỡng `0.20`, dù chiếc bàn nhìn thấy rất rõ. Vì vậy ngưỡng chỉ lọc prediction, không phải quy tắc để bỏ vật thể khi tạo ground truth.
5. **Quy tắc box:** mỗi vật thể thuộc taxonomy có một box riêng, ôm sát phần nhìn thấy của vật. Không vẽ phần bị che hoặc phần nằm ngoài ảnh; nếu vật bị cắt ở mép ảnh thì cạnh box dừng ở mép đó. Vật vẫn được gán nhãn nếu còn nhận ra được, bất kể model cho score bao nhiêu.
6. **Vật bị cắt mép:** record `person`, score 0.610931, `bbox_xyxy`=[0.12, 263.18, 61.55, 310.78] nằm sát mép trái. Trên PNG chỉ thấy một phần cánh tay và bàn tay. Tôi sẽ **viết thành quy tắc trong guideline** vì vật bị cắt mép là tình huống lặp lại ở nhiều ảnh và có thể xử lý nhất quán: nếu còn nhận ra là `person` thì vẽ box cho phần nhìn thấy. Tôi chỉ escalation khi phần nhìn thấy quá nhỏ hoặc không đủ để xác định đó là vật gì.

## 3. Phân đoạn theo từng đối tượng

1. **Record đã chọn:** `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; `polygon_point_count`=348 ; 5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]
2. Box của record này chỉ là một hình chữ nhật `[385.45, 66.44, 498.02, 348.58]`, nên chứa cả phần nền và nồi chảo treo phía sau. Polygon có 348 điểm bám theo đường viền của người, nên cho thấy hình dạng đầu, thân, tay và chân, đồng thời tách người khỏi phần lớn nền xung quanh.
3. `instance_id` dùng để phân biệt từng đối tượng trong output lab. Ví dụ `kitchen-002`, `kitchen-003` và `kitchen-006` đều là `bowl` (`class_id` 45) nhưng là ba instance khác nhau. Nó không phải class ID vì class ID cho biết vật thuộc lớp nào. Nó cũng không phải tracking ID vì không dùng để theo dõi cùng một vật qua nhiều khung hình của video.
4. **Quy tắc vẽ biên mask:** mỗi instance có một polygon riêng, bám sát biên phần nhìn thấy. Không tính nền, không tính vật khác và không đoán phần bị che. Khi hai vật chồng lên nhau, vùng chồng thuộc về vật nằm phía trước.
5. **Chỗ khó:** `kitchen-002` (`bowl`, score 0.735744, box [31.44, 343.77, 100.89, 385.55]) và `kitchen-006` (`bowl`, score 0.528201, box [0.11, 302.85, 85.7, 370.15]) có box chồng lên nhau ở góc trái dưới của ảnh. Về **biên**, tôi dùng guideline vì quy tắc "vùng chồng thuộc về vật phía trước" áp dụng được cho nhiều ảnh. Về **tên lớp**, tôi escalation vì `kitchen-006` trông giống một chiếc bánh hoặc đĩa tròn hơn là bát, và tôi không nên tự quyết khi guideline chưa định nghĩa ranh giới của `bowl`.

## 4. Vòng đời và kiểm tra chất lượng

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một lớp cho cả ảnh, gồm `class_id`, `class_name`, `taxonomy_name` | `traffic`: top-1 là `cab` (0.510915), nhưng trên ảnh thấy rõ nhiều xe buýt và ô tô. Cả 5 lớp trong top-5 đều là một loại xe, không lớp nào mô tả cả cảnh giao thông | Đọc guideline, chọn một lớp theo chủ đề chính; escalation nếu không có lớp phù hợp | Lớp có đúng guideline và đúng taxonomy không; không dựa vào score để chấp nhận nhãn |
| Phát hiện vật thể | Mỗi vật một lớp và một box pixel `[x_min, y_min, x_max, y_max]` | `oven` (0.6343, box [489.09, 200.73, 617.81, 344.88]) bao cả bồn rửa và tủ đen, chưa rõ có phải lò không. `person` (0.610931) chỉ là cánh tay bị cắt ở mép trái. Bàn bếp không có box ở ngưỡng `0.35` | Vẽ box sát phần nhìn thấy cho từng vật, gồm cả vật mà model bỏ sót; ghi lại ca khó | Thiếu vật, thừa box, sai lớp, box lỏng, box trùng; đối chiếu ảnh khi đổi ngưỡng |
| Instance segmentation | Mỗi instance một lớp và một polygon, có ID riêng | `kitchen-004`: `potted plant` (0.632235) nằm trên bó lá treo, không thấy chậu. `kitchen-005`: `dining table` (0.578416) gọi bàn sơ chế trong bếp là bàn ăn, mask phủ cả một phần vật trên bàn và thùng kim loại dưới bàn | Vẽ polygon riêng cho từng instance theo quy tắc biên; tách vật đặt trên bàn khỏi mask của bàn; escalation khi lớp chưa rõ | Biên có tràn sang vật khác không; có gộp hai instance không; lớp có đúng không; mask có khớp với ảnh không |

## 5. An toàn dữ liệu

1. Tôi chỉ dùng ba ảnh COCO công khai có sẵn trong notebook, và không đưa ảnh cá nhân, dữ liệu nội bộ hay thông tin cá nhân vào Colab, báo cáo hoặc repository công khai.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng lại và báo cho Lab Coach.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
