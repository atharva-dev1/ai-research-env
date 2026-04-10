import { create } from 'zustand'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://localhost:7860'

const TASK_META = {
  'cv-classification': { label: 'Computer Vision', color: '#6366f1', bg: '#eef2ff', icon: '🖼️', difficulty: 'Easy' },
  'nlp-sentiment':     { label: 'NLP Sentiment',   color: '#0ea5e9', bg: '#f0f9ff', icon: '💬', difficulty: 'Medium' },
  'healthcare-tabular':{ label: 'Healthcare ML',   color: '#f59e0b', bg: '#fffbeb', icon: '🏥', difficulty: 'Hard' },
}

const PHASE_COLORS = {
  read_paper:          '#8b5cf6',
  propose_hypothesis:  '#3b82f6',
  design_experiment:   '#06b6d4',
  run_experiment:      '#10b981',
  analyze_results:     '#f59e0b',
  refine_hypothesis:   '#ef4444',
  final_answer:        '#ec4899',
}

export const useStore = create((set, get) => ({
  // Session
  sessionId: null,
  taskName: 'cv-classification',
  observation: null,
  done: false,
  error: null,
  loading: false,

  // History
  history: [],        // [{step, action_type, content, reward, done, info}]
  rewardHistory: [],  // [{step, reward, cumulative}]

  // Current action form
  actionType: 'read_paper',
  actionContent: '',

  // Meta
  taskMeta: TASK_META,
  phaseColors: PHASE_COLORS,

  setTaskName: (t) => set({ taskName: t }),
  setActionType: (a) => set({ actionType: a }),
  setActionContent: (c) => set({ actionContent: c }),

  reset: async (taskName) => {
    set({ loading: true, error: null, history: [], rewardHistory: [], done: false })
    try {
      const { data } = await axios.post(`${API}/reset`, { task_name: taskName || get().taskName })
      set({
        sessionId: data.session_id,
        observation: data.observation,
        taskName: data.observation.task_name,
        actionType: data.observation.allowed_actions[0] || 'read_paper',
        loading: false,
      })
    } catch (e) {
      set({ error: e.message, loading: false })
    }
  },

  step: async () => {
    const { sessionId, actionType, actionContent, history, rewardHistory } = get()
    if (!sessionId || !actionContent.trim()) return
    set({ loading: true, error: null })
    try {
      const { data } = await axios.post(`${API}/step`, {
        session_id: sessionId,
        action: { action_type: actionType, content: actionContent },
      })
      const cumulative = (rewardHistory.at(-1)?.cumulative || 0) + data.reward
      const newEntry = {
        step: data.observation.step_number,
        action_type: actionType,
        content: actionContent,
        reward: data.reward,
        done: data.done,
        info: data.info,
        feedback: data.info.grader_feedback,
      }
      set({
        observation: data.observation,
        done: data.done,
        history: [...history, newEntry],
        rewardHistory: [...rewardHistory, {
          step: data.observation.step_number,
          reward: +data.reward.toFixed(3),
          cumulative: +cumulative.toFixed(3),
          raw: +(data.info.raw_score || 0).toFixed(3),
        }],
        actionContent: '',
        actionType: data.observation.allowed_actions[0] || actionType,
        loading: false,
      })
    } catch (e) {
      set({ error: e.message, loading: false })
    }
  },
}))
