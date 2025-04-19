from flask import Flask, render_template, request, jsonify, redirect, url_for
import os, json, numpy as np
from sentence_transformers import SentenceTransformer
import random

app = Flask(__name__)

total_attempts = 0
sequential_mode = False
DATA_FOLDER = './data'
SIMILARITY_THRESHOLD = 0.8

cards = []             # 전체 카드 리스트 (파일에서 로드)
current_file = None    # 현재 선택된 파일
current_cycle = []     # 현재 사이클: 아직 맞추지 않은 카드들
completed_cycle = []   # 맞춘 카드들

listen_mode = False



# 코사인 유사도 함수
def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@app.route('/set_listen_mode', methods=['POST'])
def set_listen_mode():
    global listen_mode
    global listen_mode_card_cnt
    listen_mode_card_cnt = 0
    listen_mode = request.json.get('listen_mode', False)
    return jsonify({'status': 'ok'})

@app.route('/')
def subject_selector():
    subjects = [d for d in os.listdir(DATA_FOLDER) if os.path.isdir(os.path.join(DATA_FOLDER, d))]
    return render_template('select_subject.html', subjects=subjects)

@app.route('/stats')
def show_stats():
    total = len(cards)
    completed = len(completed_cycle)
    remaining = len(current_cycle)
    accuracy = (completed / total * 100) if total > 0 else 0

    mistake_rate = ((total_attempts - completed) / total_attempts * 100) if total_attempts > 0 else 0
    true_accuracy = (completed / total_attempts * 100) if total_attempts > 0 else 0

    return render_template('stats.html', 
                           total=total,
                           completed=completed,
                           remaining=remaining,
                           accuracy=round(true_accuracy, 2),      # 총 시도 대비 정답률
                           mistake_rate=round(mistake_rate, 2))   # 총 시도 대비 오답률




@app.route('/edit_card/<subject>/<filename>', methods=['GET', 'POST'])
def edit_card(subject, filename):
    full_path = os.path.join(DATA_FOLDER, subject, filename + '.json')
    if not os.path.exists(full_path):
        return "파일이 존재하지 않습니다.", 404

    if request.method == 'POST':
        original_question = request.form.get('original_question')
        new_question = request.form.get('question')
        new_answer = request.form.get('answer')

        forbidden_chars = ['\n', '"', '\\']
        if any(c in new_question for c in forbidden_chars) or any(c in new_answer for c in forbidden_chars):
            return "질문이나 답변에 사용할 수 없는 문자가 포함되어 있습니다.", 400

        try:
            with open(full_path, 'r+', encoding='utf-8') as f:
                data = json.load(f)
                for card in data:
                    if card['question'] == original_question:
                        card['question'] = new_question
                        card['answer'] = new_answer
                        break
                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=4)
                f.truncate()
        except Exception as e:
            return f"수정 중 오류 발생: {e}", 500

        return redirect(url_for('edit_card', subject=subject, filename=filename))

    with open(full_path, 'r', encoding='utf-8') as f:
        cards = json.load(f)

    return render_template('edit_card.html', subject=subject, filename=filename, cards=cards)



@app.route('/new_set', methods=['GET', 'POST'])
def new_set():
    if request.method == 'POST':
        mode = request.form.get('mode')

        if mode == 'create':
            subject = request.form.get('subject')
            filename = request.form.get('filename')
            if not subject or not filename:
                return "입력값 부족!", 400

            folder_path = os.path.join(DATA_FOLDER, subject)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            full_path = os.path.join(folder_path, filename + '.json')

            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=4)

            return redirect(url_for('add_card', subject=subject, filename=filename))

        elif mode == 'edit':
            subject = request.form.get('edit_subject')
            filename = request.form.get('existing_set')
            if not subject or not filename:
                return "기존 세트 선택 오류!", 400

            return redirect(url_for('add_card', subject=subject, filename=filename))

        else:
            return "잘못된 모드!", 400

    # GET 요청
    subjects = [d for d in os.listdir(DATA_FOLDER) if os.path.isdir(os.path.join(DATA_FOLDER, d))]

    sets_by_subject = {}
    for subject in subjects:
        subject_path = os.path.join(DATA_FOLDER, subject)
        sets = []
        for file in os.listdir(subject_path):
            if file.endswith(".json"):
                sets.append(file[:-5])  # .json 제거
        sets_by_subject[subject] = sets

    return render_template('new_set.html', subjects=subjects, sets_by_subject=sets_by_subject)




@app.route('/set_sequential_mode', methods=['POST'])
def set_sequential_mode():
    mode = request.form.get('sequential')
    global sequential_mode
    
    mode = request.form.get('sequential') == 'on'
    if mode:
        sequential_mode = True
    else:
        sequential_mode = False
    return redirect(url_for('subject_selector'))

@app.route('/add_card/<subject>/<filename>', methods=['GET', 'POST'])
def add_card(subject, filename):
    full_path = os.path.join(DATA_FOLDER, subject, filename + '.json')
    if request.method == 'POST':
        question = request.form.get('question')
        answer = request.form.get('answer')

        # 특수문자 필터링
        forbidden_chars = ['\n', '"', '\\']
        if any(c in question for c in forbidden_chars) or any(c in answer for c in forbidden_chars):
            return "질문이나 답변에 사용할 수 없는 문자가 포함되어 있습니다.", 400

        if question and answer:
            try:
                with open(full_path, 'r+', encoding='utf-8') as f:
                    data = json.load(f)
                    data.append({'question': question, 'answer': answer})
                    f.seek(0)
                    json.dump(data, f, ensure_ascii=False, indent=4)
                    f.truncate()
            except Exception as e:
                return f"파일 저장 중 오류 발생: {e}", 500
        return redirect(url_for('add_card', subject=subject, filename=filename))
    
    return render_template('add_card.html', subject=subject, filename=filename)


@app.route('/delete_card/<subject>/<filename>', methods=['GET', 'POST'])
def delete_card(subject, filename):
    full_path = os.path.join(DATA_FOLDER, subject, filename + '.json')
    if not os.path.exists(full_path):
        return "파일이 존재하지 않습니다.", 404

    # 카드 삭제 처리
    if request.method == 'POST':
        question_to_delete = request.form.get('question')
        try:
            with open(full_path, 'r+', encoding='utf-8') as f:
                data = json.load(f)
                data = [card for card in data if card['question'] != question_to_delete]
                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=4)
                f.truncate()
        except Exception as e:
            return f"삭제 중 오류 발생: {e}", 500
        return redirect(url_for('delete_card', subject=subject, filename=filename))

    # 카드 목록 보여주기
    with open(full_path, 'r', encoding='utf-8') as f:
        cards = json.load(f)

    return render_template('delete_card.html', subject=subject, filename=filename, cards=cards)




@app.route('/subject/<subject>')
def file_selector(subject):
    subject_path = os.path.join(DATA_FOLDER, subject)
    files = [f for f in os.listdir(subject_path) if f.endswith('.json')]
    return render_template('select_file.html', subject=subject, files=files)

@app.route('/load/<subject>/<filename>')
def load_cards(subject, filename):
    global cards, current_file, current_cycle, completed_cycle
    current_file = os.path.join(DATA_FOLDER, subject, filename)
    try:
        with open(current_file, 'r', encoding='utf-8') as f:
            cards = json.load(f)
        # 새로운 사이클 시작: 모든 카드를 current_cycle에 넣고, completed_cycle 초기화
        current_cycle = list(cards)
        completed_cycle = []
    except Exception as e:
        print("Error loading cards:", e)
        cards = []
        current_cycle = []
    return redirect(url_for('quiz_page'))

@app.route('/restart')
def restart_cycle():
    global cards, current_cycle, completed_cycle, total_attempts
    if cards:
        current_cycle = list(cards)
        completed_cycle = []
        total_attempts = 0  # ✅ 초기화
    return redirect(url_for('quiz_page'))


listen_mode_card_cnt = 0

@app.route('/quiz')
def quiz_page():
    global current_cycle
    # 모든 카드를 맞춘 경우
    if not current_cycle:
        return redirect(url_for('quiz_complete'))
    if "본문" in current_file or sequential_mode == True and not listen_mode:
        card = current_cycle[0]
    elif listen_mode:
        global listen_mode_card_cnt
        if listen_mode_card_cnt >= len(current_cycle):
            listen_mode_card_cnt = 0
        card = current_cycle[listen_mode_card_cnt]
        listen_mode_card_cnt += 1
    else:
        card = random.choice(current_cycle)

    
    # 템플릿에 현재 카드의 question과 answer 전달
    return render_template('quiz.html',
                           question=card['question'],
                           answer=card['answer'],
                           listen_mode=listen_mode)  # ✅ 전달

@app.route('/check_answer', methods=['POST'])
def check_answer():
    global current_cycle, completed_cycle, total_attempts
    data = request.get_json()
    user = data.get('user_answer')
    correct = data.get('correct_answer')
    question = data.get('question')
    if not user or not correct or not question:
        return jsonify({'error': 'Invalid input'}), 400

    # 인코딩
    if "본문" in current_file:
        emb_user = eng_model.encode(user)
        emb_correct = eng_model.encode(correct)
    else:
        emb_user = kor_model.encode(user)
        emb_correct = kor_model.encode(correct)
    sim = cosine_similarity(emb_user, emb_correct)
    result = sim >= SIMILARITY_THRESHOLD

    total_attempts += 1  # ✅ 시도 횟수 증가

    if result:
        # 현재 사이클에서 제거
        card_to_remove = None
        for card in current_cycle:
            if card.get('question') == question:
                card_to_remove = card
                break
        if card_to_remove:
            current_cycle.remove(card_to_remove)
            completed_cycle.append(card_to_remove)
    return jsonify({'correct': result, 'similarity': sim})

@app.route('/quiz_complete')
def quiz_complete():
    return render_template('quiz_complete.html')



if __name__ == '__main__':
    eng_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    kor_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    app.run(debug=True, host='0.0.0.0')
