# 📧 Email Triage RL Environment

A reinforcement learning environment for training AI agents to triage emails effectively.

---

## 📁 Project Structure
```
RL-ENVIRONMENT/
│
├── client/                      
│   ├── agent.py                 
│   ├── client.py                
│   └── examples/                
│       ├── run_easy.py
│       ├── run_medium.py
│       └── run_hard.py
│
├── data/                        
│   ├── generators/              
│   │   ├── email_generator.py
│   │   └── noise_injector.py
│   └── templates/               
│       ├── spam.json
│       ├── work.json
│       └── personal.json
│
├── env/                         
│   ├── core/                    
│   │   ├── environment.py       
│   │   ├── observation.py       
│   │   └── transition.py        
│   ├── memory/                  
│   │   ├── history.py           
│   │   └── user_memory.py       
│   ├── models/                  
│   │   ├── actions.py           
│   │   └── state.py             
│   ├── config.py                
│   └── simulator.py             
│
├── evaluation/                  
│   ├── benchmark.py             
│   └── metrics.py               
│
├── grader/                      
│   └── grader.py                
│
├── log_collector/               
│   ├── event_logger.py          
│   └── trajectory_logger.py     
│
├── reward/                      
│   ├── engine.py                
│   └── components/              
│       ├── correctness.py       
│       ├── efficiency.py        
│       └── safety.py            
│
├── scripts/                     
│   ├── debug_env.py             
│   ├── run_benchmark.py         
│   └── run_server.py            
│
├── server/                      
│   ├── app.py                   
│   ├── middleware.py            
│   ├── schemas.py               
│   └── routes/                  
│       ├── env_routes.py        
│       ├── grader_routes.py     
│       └── task_routes.py       
│
├── tasks/                       
│   ├── definitions/             
│   │   ├── easy.py              
│   │   ├── medium.py            
│   │   └── hard.py              
│   └── task_factory.py          
│
├── tests/                       
│   ├── test_client.py           
│   ├── test_env.py              
│   ├── test_reward.py           
│   └── test_server.py           
│
├── utils/                       
│   ├── heuristics.py            
│   ├── logger.py                
│   └── text_processing.py       
│
├── .env                         
├── main.py                      
├── openenv.yaml                 
├── README.md                    
└── requirements.txt             
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/RL-ENVIRONMENT.git
cd RL-ENVIRONMENT
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download spaCy model
```bash
python -m spacy download en_core_web_sm
```

### 5. Setup environment variables
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

---

## 🚀 Running the Server
```bash
uvicorn server.app:app --reload --port 8000
```

Server will be available at:
```
http://127.0.0.1:8000          # Root
http://127.0.0.1:8000/docs     # Interactive API docs
http://127.0.0.1:8000/health   # Health check
```

---

## 🎮 Running the Client

### Easy task
```bash
python client/examples/run_easy.py
```

### Medium task
```bash
python client/examples/run_medium.py
```

### Hard task
```bash
python client/examples/run_hard.py
```

---

## 🌐 API Endpoints

### Environment
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/env/reset` | Start new episode |
| POST | `/env/step` | Take one action |
| GET | `/env/render/{episode_id}` | Render current state |

### Tasks
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/tasks/levels` | Get all task levels |
| GET | `/tasks/{level}` | Get task info |

### Grader
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/grader/grade/{episode_id}` | Grade episode |

---

## 🎯 Task Levels

| Level | Description | Pass Threshold |
|-------|-------------|----------------|
| Easy | Simple email classification | 0.6 |
| Medium | Context aware email handling | 0.7 |
| Hard | Multi step email workflows | 0.8 |

---

## 🏆 Reward System

| Action | Reward |
|--------|--------|
| Correct action | +1.0 |
| Partial credit | +0.3 |
| Wrong action | -0.5 |
| Per step penalty | -0.01 |
| Reply to spam | -0.5 |
| Delete work email | -0.3 |
| Correctly mark spam | +0.2 |

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| fastapi | Web API framework |
| uvicorn | ASGI server |
| pydantic | Data validation |
| loguru | Logging |
| httpx | HTTP client |
| gymnasium | RL environment interface |
| numpy | Numerical operations |
| pandas | Data handling |
| spacy | NLP processing |
| python-dotenv | Environment variables |

---

## 🧪 Running Tests
```bash
pytest tests/
```

With coverage:
```bash
pytest tests/ --cov=. --cov-report=html
```

---

## 📊 Grading

Final score is calculated as:
```
final_score =
    accuracy       × 50%
    partial_credit × 20%
    efficiency     × 20%
    safety         × 10%
```

---
