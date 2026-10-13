import os
import uuid
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from model_utils import predict_image, DISPLAY_NAMES
from models import db, User, ChatSession, ChatMessage, ImageResult

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, '..', 'frontend', 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(
    __name__,
    template_folder='../frontend/templates',
    static_folder='../frontend/static'
)
app.secret_key = 'skin-disease-project-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Below this confidence (%), the UI shows a "not sure" warning. Tune after testing.
LOW_CONFIDENCE_THRESHOLD = 40

ALLOWED_CONDITIONS = set(DISPLAY_NAMES.values())
CONDITION_LIST = "\n".join(f"   - {name}" for name in DISPLAY_NAMES.values())

SYSTEM_PROMPT = f"""You are a friendly skin symptom assistant on a college project website called SkinSense.

Your job:
1. Ask the user about their skin symptoms conversationally, one question at a time (location, appearance, itching/pain, duration, triggers, changes over time, etc.)
2. Keep responses short and warm - 1-3 sentences per turn.
3. After gathering enough information (usually 4-6 exchanges), suggest a likely condition from ONLY this list:
{CONDITION_LIST}

4. When you give a suggestion, clearly say "Possible condition: [name]" on its own line, and always add: "This is not a medical diagnosis - please consult a dermatologist for confirmation."
5. If symptoms don't match any listed condition well, say so honestly and still recommend seeing a dermatologist.
6. If the likely condition is a malignant-lesion or skin-cancer category, or the user describes a mole or lesion that is changing, bleeding, or growing quickly, calmly but clearly urge them to see a dermatologist soon.
7. Never prescribe medication or treatment.
8. Stay strictly on the topic of skin symptoms - if the user goes off-topic, gently steer back.
"""

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
gemini_model = genai.GenerativeModel('gemini-flash-latest', system_instruction=SYSTEM_PROMPT)

FRIENDLY_RATE_LIMIT_MSG = "I'm getting a lot of requests right now - please wait about 15-20 seconds and try again."
FRIENDLY_GENERIC_ERROR_MSG = "Sorry, I'm having trouble responding right now. Please try again in a moment."

# Chat history is kept on the server (cookies are too small for long chats)
CHAT_HISTORIES = {}

# ---------------- PAGES ----------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image-predict')
def image_predict_page():
    return render_template('image_predict.html')

@app.route('/chatbot')
def chatbot_page():
    return render_template('chatbot.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

# ---------------- AUTH ----------------

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        if not username or not email or not password:
            flash('Please fill in all fields.', 'error')
            return render_template('signup.html')

        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return render_template('signup.html')

        if User.query.filter_by(username=username).first():
            flash('That username is already taken.', 'error')
            return render_template('signup.html')

        if User.query.filter_by(email=email).first():
            flash('An account with that email already exists.', 'error')
            return render_template('signup.html')

        new_user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for('index'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('index'))

        flash('Invalid username or password.', 'error')
        return render_template('login.html')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# ---------------- HISTORY ----------------

@app.route('/history')
@login_required
def history():
    chat_sessions = ChatSession.query.filter_by(user_id=current_user.id)\
        .order_by(ChatSession.created_at.desc()).all()
    image_results = ImageResult.query.filter_by(user_id=current_user.id)\
        .order_by(ImageResult.timestamp.desc()).all()
    return render_template('history.html',
                           chat_sessions=chat_sessions,
                           image_results=image_results)

# ---------------- IMAGE PREDICTION ----------------

@app.route('/predict-image', methods=['POST'])
def predict_image_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        saved_filename = None

        if current_user.is_authenticated:
            ext = os.path.splitext(secure_filename(file.filename))[1] or '.jpg'
            saved_filename = f"{uuid.uuid4().hex}{ext}"
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
            file.save(save_path)
            result = predict_image(save_path)
        else:
            result = predict_image(file)

        confidence_pct = round(result['confidence'] * 100, 2)

        if current_user.is_authenticated and saved_filename:
            record = ImageResult(
                user_id=current_user.id,
                image_filename=saved_filename,
                predicted_class=result['predicted_class'],
                confidence=confidence_pct
            )
            db.session.add(record)
            db.session.commit()

        return jsonify({
            'predicted_class': result['predicted_class'],
            'confidence': confidence_pct,
            'low_confidence': confidence_pct < LOW_CONFIDENCE_THRESHOLD,
            'urgent': result['urgent'],
            'top3': result['top3'],
            'gradcam': result['gradcam'],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ---------------- CHATBOT ----------------

def build_first_message(image_context):
    msg = "Start the conversation: greet me briefly and ask your first question about my skin symptoms."
    if isinstance(image_context, dict):
        cond = str(image_context.get('condition', ''))
        if cond in ALLOWED_CONDITIONS:
            try:
                conf = float(image_context.get('confidence', 0))
            except (TypeError, ValueError):
                conf = 0
            msg = (
                f"I just uploaded a skin photo and an image classifier predicted '{cond}' "
                f"with {conf:.0f}% confidence (it may be wrong). Greet me briefly, mention this only as a "
                f"possibility, and ask follow-up questions about my symptoms to check whether they fit."
            )
    return msg

def save_message(sender, text):
    """Save a chat message to the database if the user is logged in."""
    if not current_user.is_authenticated:
        return
    db_session_id = session.get('db_session_id')
    if not db_session_id:
        return
    db.session.add(ChatMessage(session_id=db_session_id, sender=sender, message=text))
    db.session.commit()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json() or {}
    action = data.get('action', '')

    if action == 'start':
        chat_id = uuid.uuid4().hex
        session['chat_id'] = chat_id
        CHAT_HISTORIES[chat_id] = []
        session.pop('db_session_id', None)

        if current_user.is_authenticated:
            new_session = ChatSession(user_id=current_user.id)
            db.session.add(new_session)
            db.session.commit()
            session['db_session_id'] = new_session.id

        first_msg = build_first_message(data.get('image_context'))
        try:
            chat_obj = gemini_model.start_chat(history=[])
            response = chat_obj.send_message(first_msg)
            bot_reply = response.text
            CHAT_HISTORIES[chat_id] = [
                {'role': 'user', 'parts': [first_msg]},
                {'role': 'model', 'parts': [bot_reply]},
            ]
            save_message('bot', bot_reply)
            return jsonify({'bot_message': bot_reply})
        except ResourceExhausted:
            return jsonify({'bot_message': FRIENDLY_RATE_LIMIT_MSG}), 200
        except Exception as e:
            print("Chat start error:", e)
            return jsonify({'bot_message': FRIENDLY_GENERIC_ERROR_MSG}), 200

    user_message = data.get('message', '').strip()
    if not user_message:
        return jsonify({'error': 'Empty message'}), 400

    chat_id = session.get('chat_id')
    if not chat_id or chat_id not in CHAT_HISTORIES:
        chat_id = uuid.uuid4().hex
        session['chat_id'] = chat_id
        CHAT_HISTORIES[chat_id] = []
    history_list = CHAT_HISTORIES[chat_id]

    try:
        chat_obj = gemini_model.start_chat(history=history_list)
        response = chat_obj.send_message(user_message)
        bot_reply = response.text

        history_list.append({'role': 'user', 'parts': [user_message]})
        history_list.append({'role': 'model', 'parts': [bot_reply]})

        save_message('user', user_message)
        save_message('bot', bot_reply)
        return jsonify({'bot_message': bot_reply})
    except ResourceExhausted:
        return jsonify({'bot_message': FRIENDLY_RATE_LIMIT_MSG}), 200
    except Exception as e:
        print("Chat error:", e)
        return jsonify({'bot_message': FRIENDLY_GENERIC_ERROR_MSG}), 200

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, use_reloader=False, port=5000)