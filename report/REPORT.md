# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 11/6/2026

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:**Python: 3.13.18 PyTorch: 2.11.0+cpu Ultralytics: 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

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

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
- tôi nghĩ record này đang nói cả bức ảnh vì, nó không hề có thông tin tọa độ. vì vậy tôi nghĩ rằng cab là thể hiện lớp mà model cho rằng là phù hợp nhất
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
- danh sách class không phải do model tự nghĩ ra vì trong code có :
  "name = result.names"
  trong đó biến names chứa ánh xạ giữa class_id và class_name, Khi inference, model chỉ dự đoán trong phạm vi các lớp mà nó đã được huấn luyện, không tự tạo ra những lớp mới.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
- tôi nghĩ điều này sẽ đảm bảo tính định danh, đối chiếu chính xác của dữ liệu
class_id là mã định danh của các lớp , giúp xử lý đối chiếu 1 cách chính xác
class_name sẽ giúp chúng ta nhận ra nội dung được phân loại
taxonomy_name giúp xác định hệ thống phân loại đang sử dụng
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
tôi nghĩ guideline cần nói rõ các điều sau:
- xác định đối tương, chủ đề mà bức ảnh muốn thể hiện
- ưu tiên chủ thể nổi bật hoặc quan trọng trong bức ảnh
- không lựa chọn lớp chỉ vì đối tượng đó xuất hiện trong ảnh
- nếu có đối tượng có mức độ quan trọng hay nổi bật ngang nhau thì nên có quy tắc rằng trong class đó sẽ có nội dung "không thể sác định 1 lớp duy nhất" trong class_name
6. Vì sao không được lấy `score` của model làm ground truth?
vì score trong output chỉ thể hiện độ tin cậy của model đối với 1 lớp chứ không chứng minh đó là label chính xác của ảnh


## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=bus ; `score`=0.912557 ; `bbox_xyxy`=[93.17, 187.95, 223.01, 320.91] ; `bbox_width`=129.84 ; `bbox_height`=132.96
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
x_min: vị trí cạnh trái của hộp.
y_min: vị trí cạnh trên của hộp.
x_max: vị trí cạnh phải của hộp.
y_max: vị trí cạnh dưới của hộp.
ví dụ nếu box có bbox_xyxy = [93.17, 187.95, 223.01, 320.91]
Góc trên bên trái nằm tại tọa độ (93.17, 187.95).
Góc dưới bên phải nằm tại tọa độ (223.01, 320.91).
Chiều rộng: 223.01 - 93.17 = 129.84 pixel.
Chiều cao: 320.91 - 187.95 = 132.96 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
- 2 ngưỡng 0.20 và 0.60 lần lượt có prediction là 17 và 6
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
- nếu hạ ngưỡng hay giữ nhiều hộp hơn sẽ giữ bao gồm các box không đáng tin cậy. lúc đó người review cần phải box có thực sự bao quanh vật thể hay không, kiểm tra class có chính xác hay không, xem có các bounding box có trùng 1 vật thể hay không....
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
- Annotator cần vẽ bounding box bao quanh toàn bộ phần vật thể nhìn thấy được trong ảnh, đảm bảo box sát với ranh giới vật thể nhất có thể và hạn chế tối đa phần background nằm bên trong box.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
- trên ảnh có các bowl và cup bị che khuất
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`= 
      [
        446.0,
        70.0
      ],
      [
        445.0,
        71.0
      ],
      [
        444.0,
        71.0
      ],
      [
        443.0,
        72.0
      ],
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
- polygon sử dụng nhiều điểm tọa độ [x, y] để mô tả đường biên của vật thể. nhờ vậy, polygon thể hiện hình dạng và đường viền thực tế của vật thể chi tiết hơn, kể cả những vật thể có hình dạng phức tạp hoặc không vuông vắn.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
- instance_id là mã định danh riêng cho từng đối tượng cụ thể trong output, giúp phân biệt các instance ngay cả khi chúng có cùng class_name
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
- Annotator phải vẽ đường biên mask bám sát ranh giới thực tế của vật thể, bao gồm toàn bộ phần vật thể nhìn thấy được và không bao gồm background bên ngoài. Khi vật thể có hình dạng cong hoặc phức tạp, cần sử dụng đủ điểm polygon để mô tả chính xác đường viền, tránh vẽ quá thô hoặc tạo ra các góc không đúng với hình dạng thực tế
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
ở chỗ bowl và cup tôi sẽ dùng escalation 

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Nhãn class của toàn bộ ảnh | Khó xác định chủ thể chính khi có nhiều đối tượng | Chọn class đúng theo guideline. | Kiểm tra nhãn có đúng nội dung ảnh không. |
| Phát hiện vật thể | Class + Bounding Box | Box chồng lấn, sai vị trí | Vẽ box sát vật thể và chọn đúng class | Kiểm tra class, vị trí box và vật thể bị bỏ sót |
| Instance segmentation | Class + Mask/Polygon riêng cho từng instance | Biên mờ, vật thể chạm nhau hoặc bị che khuất | Vẽ mask sát biên từng vật thể | Kiểm tra độ chính xác của mask và từng instance |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
-Tôi chỉ sử dụng dữ liệu được cung cấp cho mục đích thực hiện bài lab, không tự ý chia sẻ, tải lên nền tảng bên ngoài hoặc tiết lộ thông tin nhạy cảm có trong dữ liệu
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
-Tôi sẽ dừng xử lý dữ liệu đó, không tự ý chỉnh sửa hoặc chia sẻ, đồng thời báo cáo cho Lab Coach hoặc người phụ trách dữ liệu để được hướng dẫn cách xử lý phù hợp.

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
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
