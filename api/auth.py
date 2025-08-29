from flask import Blueprint, request, jsonify
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, get_jwt_identity, get_jwt
)
from datetime import timedelta, datetime
import os

auth_bp = Blueprint("auth", __name__)

# ตัวอย่าง user (ใน production ควรใช้ database และ hash password)
USERS = {
    "admin": os.environ.get('ADMIN_PASSWORD', '1234'),
    "user": os.environ.get('USER_PASSWORD', 'password'),
    "guest": os.environ.get('GUEST_PASSWORD', 'guest123')
}

print("DEBUG Admin Password:", os.environ.get("ADMIN_PASSWORD"))
print("DEBUG User Password:", os.environ.get("USER_PASSWORD"))
print("DEBUG Guest Password:", os.environ.get("GUEST_PASSWORD"))


# เก็บ failed login attempts (ป้องกัน brute force)
failed_attempts = {}
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_TIME = 300  # 5 นาที

def is_account_locked(username):
    """ตรวจสอบว่า account ถูกล็อคหรือไม่"""
    if username not in failed_attempts:
        return False
    
    attempts, last_attempt = failed_attempts[username]
    if attempts >= MAX_FAILED_ATTEMPTS:
        time_diff = datetime.utcnow() - last_attempt
        if time_diff.total_seconds() < LOCKOUT_TIME:
            return True
        else:
            # ล็อคหมดแล้ว รีเซ็ต
            del failed_attempts[username]
            return False
    return False

def record_failed_attempt(username):
    """บันทึกการ login ผิด"""
    if username in failed_attempts:
        attempts, _ = failed_attempts[username]
        failed_attempts[username] = (attempts + 1, datetime.utcnow())
    else:
        failed_attempts[username] = (1, datetime.utcnow())

def reset_failed_attempts(username):
    """รีเซ็ตการ login ผิดเมื่อ login สำเร็จ"""
    if username in failed_attempts:
        del failed_attempts[username]

# Login route
@auth_bp.route("/login", methods=["POST"])
def login():
    try:
        # ตรวจสอบ Content-Type - รองรับทั้ง JSON และ form data
        if request.is_json:
            data = request.get_json()
        elif request.content_type == 'application/x-www-form-urlencoded':
            data = request.form.to_dict()
        else:
            # รองรับ request ที่ไม่มี Content-Type ที่ชัดเจน
            try:
                data = request.get_json(force=True)
            except:
                data = request.form.to_dict() if request.form else {}
        
        if not data:
            return jsonify({
                "error": "ไม่มีข้อมูล",
                "code": "NO_DATA"
            }), 400
            
        username = str(data.get("username", "")).strip()
        password = str(data.get("password", ""))

        if not username or not password:
            return jsonify({
                "error": "กรุณาระบุ username และ password",
                "code": "MISSING_CREDENTIALS"
            }), 400

        # ตรวจสอบว่า account ถูกล็อคหรือไม่
        if is_account_locked(username):
            remaining_time = LOCKOUT_TIME
            if username in failed_attempts:
                attempts, last_attempt = failed_attempts[username]
                elapsed = (datetime.utcnow() - last_attempt).total_seconds()
                remaining_time = max(0, LOCKOUT_TIME - elapsed)
            
            return jsonify({
                "error": f"Account ถูกล็อคเนื่องจาก login ผิดหลายครั้ง กรุณารอ {int(remaining_time//60)} นาที {int(remaining_time%60)} วินาที",
                "code": "ACCOUNT_LOCKED",
                "retry_after": int(remaining_time)
            }), 429

        # ตรวจสอบ username และ password
        if username in USERS and USERS[username] == password:
            # Login สำเร็จ - รีเซ็ต failed attempts
            reset_failed_attempts(username)
            
            # สร้าง access token
            access_token = create_access_token(
                identity=username,
                expires_delta=timedelta(minutes=30)  # หมดอายุใน 30 นาที
            )
            
            return jsonify({
                "access_token": access_token,
                "token_type": "Bearer",
                "expires_in": 1800,  # 30 นาที (วินาที)
                "user": username,
                "login_time": datetime.utcnow().isoformat() + "Z",
                "message": f"เข้าสู่ระบบสำเร็จ ยินดีต้อนรับ {username}",
                "success": True
            }), 200

        # Login ไม่สำเร็จ - บันทึก failed attempt
        record_failed_attempt(username)
        attempts = failed_attempts.get(username, (0, None))[0]
        remaining_attempts = MAX_FAILED_ATTEMPTS - attempts
        
        response_data = {
            "error": "username หรือ password ไม่ถูกต้อง",
            "code": "INVALID_CREDENTIALS",
            "remaining_attempts": max(0, remaining_attempts),
            "success": False
        }
        
        if remaining_attempts > 0:
            response_data["warning"] = f"คำเตือน: เหลือโอกาสในการ login อีก {remaining_attempts} ครั้ง"
        else:
            response_data["warning"] = f"Account จะถูกล็อคเป็นเวลา {LOCKOUT_TIME//60} นาที"
        
        return jsonify(response_data), 401

    except Exception as e:
        # Log error สำหรับ debugging
        print(f"❌ Error in login: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        
        return jsonify({
            "error": "เกิดข้อผิดพลาดภายในเซิร์ฟเวอร์",
            "code": "INTERNAL_SERVER_ERROR",
            "success": False
        }), 500

# Get user info route
@auth_bp.route("/me", methods=["GET"])
@jwt_required()
def get_user_info():
    """ดึงข้อมูลผู้ใช้ปัจจุบัน"""
    try:
        current_user = get_jwt_identity()
        token_data = get_jwt()
        
        return jsonify({
            "user": current_user,
            "token_issued_at": datetime.fromtimestamp(token_data['iat']).isoformat() + "Z",
            "token_expires_at": datetime.fromtimestamp(token_data['exp']).isoformat() + "Z",
            "token_id": token_data.get('jti', 'unknown'),
            "success": True
        }), 200
    except Exception as e:
        print(f"❌ Error in get_user_info: {str(e)}")
        return jsonify({
            "error": "ไม่สามารถดึงข้อมูลผู้ใช้ได้",
            "code": "USER_INFO_ERROR",
            "success": False
        }), 500

# Change password route  
@auth_bp.route("/change-password", methods=["POST"])
@jwt_required()
def change_password():
    """เปลี่ยนรหัสผ่าน"""
    try:
        current_user = get_jwt_identity()
        
        # รองรับทั้ง JSON และ form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        if not data:
            return jsonify({
                "error": "ไม่มีข้อมูล",
                "code": "NO_DATA",
                "success": False
            }), 400
        
        old_password = str(data.get("old_password", ""))
        new_password = str(data.get("new_password", ""))
        
        if not old_password or not new_password:
            return jsonify({
                "error": "กรุณาระบุรหัสผ่านเก่าและใหม่",
                "code": "MISSING_PASSWORDS",
                "success": False
            }), 400
        
        # ตรวจสอบรหัสผ่านเก่า
        if USERS.get(current_user) != old_password:
            return jsonify({
                "error": "รหัสผ่านเก่าไม่ถูกต้อง",
                "code": "INVALID_OLD_PASSWORD",
                "success": False
            }), 401
        
        # ตรวจสอบรหัสผ่านใหม่
        if len(new_password) < 4:
            return jsonify({
                "error": "รหัสผ่านใหม่ต้องมีอย่างน้อย 4 ตัวอักษร",
                "code": "PASSWORD_TOO_SHORT",
                "success": False
            }), 400
        
        if old_password == new_password:
            return jsonify({
                "error": "รหัสผ่านใหม่ต้องแตกต่างจากรหัสผ่านเก่า",
                "code": "SAME_PASSWORD",
                "success": False
            }), 400
        
        # เปลี่ยนรหัสผ่าน
        USERS[current_user] = new_password
        
        return jsonify({
            "message": "เปลี่ยนรหัสผ่านสำเร็จ",
            "user": current_user,
            "changed_at": datetime.utcnow().isoformat() + "Z",
            "warning": "การเปลี่ยนแปลงนี้เป็นชั่วคราว (in-memory only)",
            "success": True
        }), 200
        
    except Exception as e:
        print(f"❌ Error in change password: {str(e)}")
        return jsonify({
            "error": "เกิดข้อผิดพลาดในการเปลี่ยนรหัสผ่าน",
            "code": "CHANGE_PASSWORD_ERROR",
            "success": False
        }), 500

# Admin route - ดูข้อมูล failed attempts
@auth_bp.route("/admin/failed-attempts", methods=["GET"])
@jwt_required()
def get_failed_attempts():
    """ดูข้อมูล failed login attempts (admin เท่านั้น)"""
    try:
        current_user = get_jwt_identity()
        if current_user != "admin":
            return jsonify({
                "error": "ไม่มีสิทธิ์เข้าถึง",
                "code": "ACCESS_DENIED",
                "success": False
            }), 403
        
        # แปลงข้อมูลให้เป็นรูปแบบที่ส่งออกได้
        attempts_data = {}
        for username, (attempts, last_attempt) in failed_attempts.items():
            time_until_unlock = 0
            if attempts >= MAX_FAILED_ATTEMPTS:
                elapsed = (datetime.utcnow() - last_attempt).total_seconds()
                time_until_unlock = max(0, LOCKOUT_TIME - elapsed)
            
            attempts_data[username] = {
                "attempts": attempts,
                "last_attempt": last_attempt.isoformat() + "Z",
                "is_locked": is_account_locked(username),
                "time_until_unlock_seconds": int(time_until_unlock)
            }
        
        return jsonify({
            "failed_attempts": attempts_data,
            "max_attempts": MAX_FAILED_ATTEMPTS,
            "lockout_time_seconds": LOCKOUT_TIME,
            "total_locked_accounts": sum(1 for data in attempts_data.values() if data["is_locked"]),
            "accessed_by": current_user,
            "accessed_at": datetime.utcnow().isoformat() + "Z",
            "success": True
        }), 200
        
    except Exception as e:
        print(f"❌ Error in get failed attempts: {str(e)}")
        return jsonify({
            "error": "เกิดข้อผิดพลาดในการดึงข้อมูล",
            "code": "GET_ATTEMPTS_ERROR",
            "success": False
        }), 500

# Admin route - ปลดล็อค user
@auth_bp.route("/admin/unlock-user", methods=["POST"])
@jwt_required()
def unlock_user():
    """ปลดล็อค user (admin เท่านั้น)"""
    try:
        current_user = get_jwt_identity()
        if current_user != "admin":
            return jsonify({
                "error": "ไม่มีสิทธิ์เข้าถึง",
                "code": "ACCESS_DENIED",
                "success": False
            }), 403
        
        # รองรับทั้ง JSON และ form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        if not data:
            return jsonify({
                "error": "ไม่มีข้อมูล",
                "code": "NO_DATA",
                "success": False
            }), 400
        
        username_to_unlock = str(data.get("username", "")).strip()
        if not username_to_unlock:
            return jsonify({
                "error": "กรุณาระบุ username ที่ต้องการปลดล็อค",
                "code": "MISSING_USERNAME",
                "success": False
            }), 400
        
        if username_to_unlock not in failed_attempts:
            return jsonify({
                "error": f"ไม่พบ failed attempts สำหรับ username: {username_to_unlock}",
                "code": "USER_NOT_FOUND",
                "success": False
            }), 404
        
        # ปลดล็อคด้วยการลบ failed attempts
        del failed_attempts[username_to_unlock]
        
        return jsonify({
            "message": f"ปลดล็อค user '{username_to_unlock}' สำเร็จ",
            "unlocked_user": username_to_unlock,
            "unlocked_by": current_user,
            "unlocked_at": datetime.utcnow().isoformat() + "Z",
            "success": True
        }), 200
        
    except Exception as e:
        print(f"❌ Error in unlock user: {str(e)}")
        return jsonify({
            "error": "เกิดข้อผิดพลาดในการปลดล็อค user",
            "code": "UNLOCK_USER_ERROR",
            "success": False
        }), 500

# Health check สำหรับ auth module
@auth_bp.route("/health", methods=["GET"])
def auth_health_check():
    """ตรวจสอบสถานะ auth module"""
    try:
        locked_count = sum(1 for username in failed_attempts.keys() if is_account_locked(username))
        failed_count = len(failed_attempts)
        
        return jsonify({
            "status": "healthy",
            "module": "auth",
            "users_count": len(USERS),
            "failed_attempts_count": failed_count,
            "locked_accounts": locked_count,
            "max_failed_attempts": MAX_FAILED_ATTEMPTS,
            "lockout_time_minutes": LOCKOUT_TIME // 60,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "success": True,
            "environment": "render" if os.environ.get('RENDER') else "local"
        }), 200
    except Exception as e:
        print(f"❌ Error in health check: {str(e)}")
        return jsonify({
            "status": "error",
            "module": "auth",
            "error": str(e),
            "success": False
        }), 500

# Route สำหรับทดสอบ
@auth_bp.route("/test", methods=["GET", "POST"])
def test_endpoint():
    """Endpoint สำหรับทดสอบ"""
    return jsonify({
        "message": "Auth module is working",
        "method": request.method,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "success": True
    }), 200