# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Chu Mạnh

Công cụ gán nhãn đã dùng: CVAT

Sao chép file này thành `reports/REPORT.md` rồi điền vào các chỗ cần điền. Mọi con số phải truy được
từ `reports/rounds_table.md`, `outputs/selection_round1.csv`, `outputs/metrics_round*.json` hoặc
`outputs/round*_diff.md`. Không coi nhãn test do mô hình tạo là chân lý tuyệt đối.

## 1. Dữ liệu và cách chia tập

Tại sao tập chưa gán nhãn (pool) và tập kiểm thử (test set) được chia theo trục thời gian, có vùng
đệm ở giữa, thay vì chia ngẫu nhiên? Nếu chia ngẫu nhiên, số đo trên tập kiểm thử sẽ bị lệch theo
hướng nào, và vì sao?

Tập chưa gán nhãn và tập kiểm thử được chia theo trục thời gian và có vùng đệm ở giữa là để tránh data ở pool và test giống nhau quá do frame ở sát nhau.

Nếu chia ngẫu nhiên, bạn sẽ rất dễ đưa các frame gần nhau, cùng cảnh, cùng góc máy, cùng luồng giao thông vào cả train và test. Khi đó:

* model đã thấy “bản sao gần như nhau” ở train
* lúc test nó chỉ cần nhận diện lại hình rất tương tự
* số đo sẽ “ảo” cao hơn thực tế

## 2. Mô hình khởi đầu lạnh (cold start)

Chép dòng vòng 0 từ `rounds_table.md`. Dựa vào `outputs/compare_round0.jpg`, cho biết mô hình khởi
đầu lạnh không khớp nhãn tham chiếu ở những loại xe nào. Độ phủ (recall) theo kích thước xe cho
thấy điều gì? Một trường hợp nào cần người rà lại nhãn tham chiếu trước khi kết luận mô hình sai?

Mô hình khởi đầu lạnh không khớp tốt nhất ở “xe nhỏ, xe xa, xe mờ, xe bị che”.

| vòng | model                                   | ảnh train | box train |  AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 |    F1 | R small | R medium | R large |
| ----: | --------------------------------------- | ---------: | --------: | ----: | --------------------: | -----: | -----: | ----: | ------: | -------: | ------: |
|     0 | yolov8n cold start (COCO car+bus+truck) |          0 |         0 | 0.771 |                    — |  0.925 |  0.489 | 0.640 |   0.182 |    0.547 |   0.561 |

Dữ liệu trong `metrics_round0.json` cho thấy:

* small recall = 0.1818
* medium recall = 0.5473
* large recall = 0.561

Mô hình khởi đầu lạnh ưu tiên phát hiện xe lớn và rõ hơn, nhưng bỏ sót nhiều xe nhỏ ở xa, đặc biệt trong nền tối và có ánh sáng chói.

Nếu hình có một xe rõ ràng nhưng nhãn tham chiếu thiếu hoặc sai, thì trước khi chê mô hình, phải rà lại label tham chiếu. Ví dụ như model dự đoán đúng xe đó nhưng hệ thống đếm thành false positive vì label tham chiếu không có box.

## 3. Chiến lược chọn mẫu

Giải thích bằng lời công thức `score = W_U·U + W_A·A + W_D·D` và vai trò của `MIN_GAP_S`.
Dẫn ba frame trong `reports/SELECTION.md` và một frame khác để chứng minh cách bạn cân nhắc
độ bất định, ảnh gần trùng và công gán nhãn. Điểm bất định có chứng minh ảnh đó sẽ cải thiện
mô hình không? Vì sao?

Trong `al_select.py`, công thức được dựng như sau:

`score = W_U·U + W_A·A + W_D·D`

với trọng số mặc định:

* `W_U = 0.5`
* `W_A = 0.3`
* `W_D = 0.2`

Nghĩa đơn giản:

* `U` = độ bất định của mô hình: mô hình “lưỡng lự” nhất ở những box có confidence gần 0.5, vì lúc đó nó chưa chắc chắn là xe thật hay không. `U` lớn thì ảnh đó có nhiều box khó.
* `A` = tỉ lệ các box “mập mờ” (ambiguous): những box có confidence nằm trong vùng 0.15–0.50. Nhiều box như vậy nghĩa là ảnh có nhiều trường hợp cần xem lại.
* `D` = độ đa dạng theo thời gian: nếu ảnh nằm rất gần một frame đã chọn trước đó, nó không mang thêm nhiều thông tin mới; `D` thấp khi ảnh gần trùng. Nếu cách xa, `D` cao.

Trong `al_select.py`, hệ thống chọn frame theo thứ tự điểm cao, nhưng bỏ qua các frame cách nhau quá ít thời gian:

* `min_gap_s = 2.0` giây trong mặc định,
* nếu hai frame rất gần nhau, chỉ chọn 1 trong 2,
* vì cả hai gần như cùng một cảnh, gán nhãn cả hai tốn công mà học thêm ít.

Đây là chính là “ảnh gần trùng” mà bài báo cáo nhắc đến.
Nói cách khác, `MIN_GAP_S` là cơ chế “không chọn hàng loạt frame cùng một cảnh” để tránh lãng phí ngân sách rà nhãn.

Trong `SELECTION.md`, ba frame rõ nhất là:

* `frame_0182.jpg` — score 0.9591, U = 0.9182, n_ambiguous = 18
  Đây là cảnh đông xe, nhiều xe chồng/che khuất, rất dễ model bỏ sót hoặc box không chắc chắn.
* `frame_0369.jpg` — score 0.9324, U = 0.9315, n_ambiguous = 16
  Nhiều xe cùng lúc, ánh sáng tối và bóng đen, nên mô hình dễ thiếu box hoặc dính xe với nhau.
* `frame_0380.jpg` — score 0.9170, U = 0.9340, n_ambiguous = 15
  Cảnh tương tự nhưng có ánh đèn và xe mờ ở xa, rất thích hợp để sửa label vì model dễ bỏ xe ở mép ảnh hoặc xe nhỏ.

Một frame không chọn nhưng đáng chú ý:

* `frame_0372.jpg` — score 0.9101, hạng 6
  Nó có điểm cao nhưng mô tả trong báo cáo nói rất rõ: nó gần với `frame_0369.jpg` và `frame_0380.jpg`, cùng kiểu đường cao tốc, xe đông, đèn sáng, nên gần như cùng một “thể loại cảnh”. Nếu chọn thêm nó thì tốn công mà giá trị học tập không tăng nhiều. Vì vậy bị loại vì ảnh gần trùng.

Câu trả lời đúng là: không chắc chắn 100%.

Vì sao:

* Nếu ảnh có `U` cao và `A` cao, nghĩa là model đang bối rối ở nhiều box.
* Những box đó là những mẫu “đắt giá”: model chưa chắc nên rất có khả năng học được nhiều từ việc sửa nhãn.
* Nhưng không phải cứ `U` cao là chắc chắn sẽ cải thiện mô hình, vì:
  * nếu ảnh gần trùng với ảnh đã chọn, giá trị học tập thấp,
  * nếu ảnh quá khó và không rõ thực sự là xe, chỉnh sửa nhãn có thể không giúp nhiều,
  * nếu nhãn tham chiếu cũng có thể sai, cần rà lại trước khi kết luận.

## 4. Các vòng học chủ động (active learning)

Chép bảng từ `rounds_table.md`. Với mỗi vòng, trình bày:

| vòng | model                                   | ảnh train | box train |  AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 |    F1 | R small | R medium | R large |
| ----: | --------------------------------------- | ---------: | --------: | ----: | --------------------: | -----: | -----: | ----: | ------: | -------: | ------: |
|     0 | yolov8n cold start (COCO car+bus+truck) |          0 |         0 | 0.771 |                    — |  0.925 |  0.489 | 0.640 |   0.182 |    0.547 |   0.561 |

Mức độ bạn đã sửa nhãn gợi ý (số box giữ nguyên, chỉnh sửa, xoá, thêm mới, lấy từ
`outputs/round*_diff.md`):

* Model đề xuất 169 box trên 12 ảnh
* Sau khi sửa còn 350 box
* accepted = 129
* edited = 22
* deleted = 18
* added = 199
* accept rate = 76%

AP50 thay đổi bao nhiêu so với khởi đầu lạnh và so với vòng trước:

Trong reports/rounds_table.md:

* cold start: AP50 = 0.7714
* round 1: AP50 = 0.7352

Vậy:

* Δ AP50 = 0.7352 - 0.7714 = -0.0362

Nói cách khác, sau fine-tune, AP50 giảm khoảng 0.036, tức là xấu hơn so với khởi đầu lạnh. Đây là điều cần ghi nhận, không phải “sửa nhãn là luôn tốt hơn”.

Nhóm xe nào tốt lên hoặc xấu đi theo số đo trên cùng tập test:

Nhiều hơn nữa, dữ liệu ở `metrics_round0.json` và `metrics_round1.json` cho thấy:

* cold start:

  * small recall = 0.1818
  * medium recall = 0.5473
  * large recall = 0.5610
* round 1:

  * small recall = 0.0000
  * medium recall = 0.1149
  * large recall = 0.4878

Kết luận:

* xe lớn vẫn còn phát hiện tốt hơn, nhưng không còn mạnh như cold start
* xe nhỏ và xe trung bình xuống rất mạnh
* model sau fine-tune có xu hướng “giữ lại những xe lớn rõ hơn”, nhưng bỏ sót nhiều xe nhỏ hoặc xe mờ

Dựa vào các ảnh `compare_round*.jpg`, chỉ ra một ca kết quả đổi sau fine-tune (tốt hơn hoặc xấu
đi), cùng lý do có thể kiểm. Dùng `BLIND_SCAN.md`, `REVIEW_LOG.csv` và `round1_diff.md` phân biệt
quan sát độc lập, lỗi pre-label đã sửa và kết quả mô hình sau train. Mô tả một ca khó theo guideline:

Một ví dụ rất rõ là trường hợp frame_0326 trong `BLIND_SCAN.md` và `round1_diff.md`:

* đánh giá độc lập của người rà trước khi xem pre-label: “36 xe”
* model đầu vòng đề xuất chỉ 15 box
* sau khi sửa, file diff cho thấy:

  * accepted = 12
  * edited = 1
  * deleted = 2
  * added = 23

Một ca khó theo guideline là xe ở rìa trái / mép khung hình

## 5. Kết luận và giới hạn

Kết quả vòng này so với cold start ra sao? Vì sao bạn dừng hoặc tiếp tục? Đề xuất hai ca còn yếu
hoặc bất định cho vòng sau, kèm chi phí rà nhãn và nguy cơ ảnh gần trùng. Tập kiểm thử chỉ 20 ảnh,
có luật bỏ qua xe quá nhỏ và nhãn tham chiếu do mô hình tạo chưa được rà thủ công; các giới hạn đó
ảnh hưởng thế nào đến kết luận? Nếu AP50 giảm, bạn sẽ kiểm tra điều gì trước khi train thêm?

Vòng 1 cho thấy fine-tune không cải thiện được hiệu suất trên tập test so với cold start: theo `rounds_table.md` và `rounds_table.md`, AP50 giảm từ 0.771 xuống 0.735, tức là giảm khoảng 0.036; recall ở ngưỡng 0.25 giảm từ 0.489 xuống 0.134, còn F1 giảm từ 0.640 xuống 0.236. Điều này cho thấy mô hình sau fine-tune vẫn bỏ sót nhiều xe khó như xe nhỏ, xe ở mép khung hình, xe mờ và xe sát nhau. Dữ liệu ở `round1_diff.md` cũng cho thấy mức độ sửa nhãn rất lớn: model đề xuất 169 box nhưng sau khi rà lại còn 350 box, với 199 box được thêm mới và 18 box bị xoá. Vì vậy, tôi không chọn tiếp tục train thêm ngay mà trước tiên cần kiểm tra lại nhãn tham chiếu, lỗi pre-label và nhóm ảnh khó, vì tập test chỉ 20 ảnh và các trường hợp khó chiếm phần lớn sai lệch.

Frame 0331 hoặc 0369 còn yếu

* Cả hai đều nằm trong top ưu tiên của `SELECTION.md` và có số box lớn, độ bất định cao, rất phù hợp để học các trường hợp xe mờ và chồng box.
* Chi phí rà nhãn: cao, vì mỗi frame có khoảng 35–40 box cần kiểm tra, và nhiều xe bị gộp hoặc che khuất.
* Nguy cơ ảnh gần trùng: cao, vì các frame 0369, 0380, 0331 có cùng kiểu cảnh và cùng thời điểm tương tự trên đường cao tốc. Nếu chọn quá nhiều frame cùng cluster, công rà nhãn tăng nhưng giá trị học tập không tăng tương xứng.

Nếu cho luật bỏ qua xe quá nhỏ và nhãn tham chiếu do mô hình tạo chưa được rà thủ công, các giới hạn được bỏ đó sẽ khiến cho con model tìm true positive sẽ dễ hơn do nó tìm được xe trung/to tốt.

Nếu AP50 giảm, điều đầu tiên cần kiểm tra là liệu sự giảm là do nhãn sai, chọn mẫu kém, hay sự thật là model chưa học được các trường hợp khó này.
