def normalize_name(name):
    return name.strip().lower()


RECOMMENDATIONS = {

    "alu": {
        "material": "Aluminum cans",
        "bin": "Thùng rác tái chế kim loại",
        "advice": "Nên rửa sạch và ép dẹt lon trước khi bỏ để tiết kiệm diện tích và hỗ trợ tái chế hiệu quả hơn.",
        "recycle": "Có thể tái chế thành lon mới hoặc các sản phẩm kim loại khác.",
        "environment": "Tái chế nhôm giúp tiết kiệm năng lượng và giảm khai thác tài nguyên tự nhiên."
    },

    "carton": {
        "material": "Carton",
        "bin": "Thùng rác tái chế giấy",
        "advice": "Gấp gọn và giữ khô ráo. Không để dính dầu mỡ hoặc thức ăn.",
        "recycle": "Có thể tái chế thành giấy carton hoặc bao bì mới.",
        "environment": "Tái chế carton giúp giảm lượng rác thải và bảo vệ rừng."
    },

    "foam_box": {
        "material": "Foam box",
        "bin": "Thùng rác sinh hoạt",
        "advice": "Hộp xốp bẩn hoặc dính thức ăn thường khó tái chế nên cần bỏ đúng nơi quy định.",
        "recycle": "Foam box rất khó tái chế trong các hệ thống thông thường.",
        "environment": "Hộp xốp mất rất nhiều năm để phân hủy và có thể gây ô nhiễm môi trường."
    },

    "milk_box": {
        "material": "Milk box",
        "bin": "Thùng rác tái chế giấy hỗn hợp",
        "advice": "Nên rửa sơ và để khô trước khi bỏ để tránh gây mùi và ảnh hưởng đến quá trình tái chế.",
        "recycle": "Có thể tái chế thành giấy tái chế hoặc vật liệu tổng hợp.",
        "environment": "Phân loại đúng hộp sữa giúp giảm lượng rác khó xử lý."
    },

    "paper": {
        "material": "Paper",
        "bin": "Thùng rác tái chế giấy",
        "advice": "Giữ giấy sạch và khô. Giấy dính nước hoặc thức ăn sẽ khó tái chế.",
        "recycle": "Có thể tái chế thành giấy mới, sổ tay hoặc thùng carton.",
        "environment": "Tái chế giấy giúp giảm chặt phá rừng và tiết kiệm tài nguyên."
    },

    "paper_cup": {
        "material": "Paper cup",
        "bin": "Thùng rác sinh hoạt",
        "advice": "Ly giấy thường có lớp phủ chống thấm bên trong nên khó tái chế như giấy thông thường.",
        "recycle": "Khó tái chế do có lớp nhựa chống thấm bên trong.",
        "environment": "Nên hạn chế sử dụng ly giấy dùng một lần để giảm rác thải."
    },

    "pet": {
        "material": "PET bottle",
        "bin": "Thùng rác tái chế nhựa",
        "advice": "Nên rửa sạch chai, bóp gọn và tháo nắp để thuận tiện cho việc tái chế.",
        "recycle": "Có thể tái chế thành sợi polyester hoặc chai nhựa mới.",
        "environment": "Tái chế nhựa giúp giảm ô nhiễm môi trường và rác thải đại dương."
    },

    "plastic_cup": {
        "material": "Clear plastic cup",
        "bin": "Thùng rác tái chế nhựa",
        "advice": "Nên làm sạch trước khi bỏ để tránh lẫn tạp chất trong quá trình tái chế.",
        "recycle": "Có thể tái chế thành các sản phẩm nhựa mới.",
        "environment": "Phân loại đúng nhựa giúp giảm lượng rác khó phân hủy."
    },

    "other": {
        "material": "Other trash",
        "bin": "Thùng rác sinh hoạt",
        "advice": "Đây là loại rác khó phân loại hoặc khó tái chế, nên bỏ vào rác sinh hoạt thông thường.",
        "recycle": "Loại rác này thường khó tái chế hoặc không thể tái chế.",
        "environment": "Cần giảm sử dụng các vật liệu khó tái chế để bảo vệ môi trường."
    }
}


def get_recommendation(class_name):
    key = normalize_name(class_name)

    if key in RECOMMENDATIONS:
        return RECOMMENDATIONS[key]

    return {
        "material": class_name,
        "bin": "Thùng rác sinh hoạt",
        "advice": "Chưa có mô tả chi tiết cho loại vật liệu này.",
        "recycle": "Chưa có thông tin tái chế.",
        "environment": "Hãy phân loại rác đúng cách để bảo vệ môi trường."
    }