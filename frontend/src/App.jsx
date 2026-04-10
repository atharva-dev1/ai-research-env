import { useEffect } from 'react'
import { useStore } from './store/useStore'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Legend
} from 'recharts'

const PHASE_ORDER = ['read_paper','propose_hypothesis','design_experiment','run_experiment','analyze_results','refine_hypothesis','final_answer']
const PHASE_LABELS = {
  read_paper:'Read Paper', propose_hypothesis:'Propose Hypothesis', design_experiment:'Design Experiment',
  run_experiment:'Run Experiment', analyze_results:'Analyze Results', refine_hypothesis:'Refine Hypothesis', final_answer:'Final Answer',
}

function Badge({ children, color='#6366f1', bg='#eef2ff' }) {
  return <span style={{background:bg,color,border:`1px solid ${color}33`,borderRadius:6,padding:'2px 10px',fontSize:12,fontWeight:600}}>{children}</span>
}

function PhaseProgress({ progress={}, current }) {
  return (
    <div style={{display:'flex',gap:4,flexWrap:'wrap'}}>
      {PHASE_ORDER.map(p=>{
        const done=progress[p], active=p===current
        return <div key={p} style={{padding:'4px 10px',borderRadius:20,fontSize:11,fontWeight:600,
          background:done?'#10b981':active?'#6366f1':'#f1f5f9',color:done||active?'#fff':'#94a3b8',
          border:`2px solid ${done?'#10b981':active?'#6366f1':'#e2e8f0'}`,transition:'all 0.3s'}}>
          {done?'✓ ':active?'▶ ':''}{PHASE_LABELS[p]}
        </div>
      })}
    </div>
  )
}

function TaskCard({ id, meta, selected, onSelect }) {
  return (
    <div onClick={()=>onSelect(id)} style={{cursor:'pointer',borderRadius:12,padding:'14px 18px',
      border:`2px solid ${selected?meta.color:'#e2e8f0'}`,background:selected?meta.bg:'#fff',
      transition:'all 0.2s',boxShadow:selected?`0 0 0 3px ${meta.color}22`:'none'}}>
      <div style={{fontSize:22}}>{meta.icon}</div>
      <div style={{fontWeight:700,fontSize:14,color:'#1e293b',marginTop:4}}>{meta.label}</div>
      <Badge color={meta.color} bg={meta.bg}>{meta.difficulty}</Badge>
    </div>
  )
}

function HistoryItem({ entry, phaseColors }) {
  const col=phaseColors[entry.action_type]||'#64748b'
  return (
    <div style={{borderLeft:`3px solid ${col}`,paddingLeft:12,marginBottom:14}}>
      <div style={{display:'flex',alignItems:'center',gap:8,marginBottom:4}}>
        <span style={{fontWeight:700,color:col,fontSize:12,textTransform:'uppercase',letterSpacing:1}}>{PHASE_LABELS[entry.action_type]}</span>
        <span style={{marginLeft:'auto',fontWeight:700,color:entry.reward>0.5?'#10b981':'#f59e0b',fontSize:13}}>+{entry.reward.toFixed(3)}</span>
        {entry.done&&<Badge color='#10b981' bg='#f0fdf4'>Done</Badge>}
      </div>
      <p style={{fontSize:13,color:'#475569',margin:'0 0 4px',lineHeight:1.5,whiteSpace:'pre-wrap'}}>
        {entry.content.length>300?entry.content.slice(0,300)+'…':entry.content}
      </p>
      {entry.feedback&&<div style={{fontSize:12,color:'#7c3aed',background:'#f5f3ff',borderRadius:6,padding:'4px 8px',marginTop:4}}>💡 {entry.feedback}</div>}
    </div>
  )
}

export default function App() {
  const { sessionId, taskName, observation:obs, done, error, loading,
    history, rewardHistory, actionType, actionContent,
    taskMeta, phaseColors, setTaskName, setActionType, setActionContent, reset, step } = useStore()

  useEffect(()=>{reset('cv-classification')},[])

  const totalReward=rewardHistory.at(-1)?.cumulative||0
  const bestRaw=Math.max(...history.map(h=>h.info?.raw_score||0),0)
  const completedPhases=obs?.progress?Object.values(obs.progress).filter(Boolean).length:0

  return (
    <div style={{minHeight:'100vh',background:'linear-gradient(135deg,#f8fafc 0%,#f1f5f9 100%)',fontFamily:"'DM Sans','Segoe UI',sans-serif"}}>
      <div style={{background:'#0f172a',padding:'16px 32px',display:'flex',alignItems:'center',gap:16}}>
        <div style={{fontSize:28}}>🔬</div>
        <div>
          <div style={{color:'#fff',fontWeight:800,fontSize:20,letterSpacing:-0.5}}>AI Research Env</div>
          <div style={{color:'#64748b',fontSize:12}}>OpenEnv-compatible Scientific Discovery Platform</div>
        </div>
        {sessionId&&<div style={{marginLeft:'auto',display:'flex',gap:12,alignItems:'center'}}>
          <div style={{color:'#94a3b8',fontSize:12}}>Session: <span style={{color:'#7dd3fc',fontFamily:'monospace'}}>{sessionId.slice(0,8)}…</span></div>
          <div style={{color:'#10b981',fontWeight:700,fontSize:16}}>⚡ {totalReward.toFixed(3)}</div>
        </div>}
      </div>

      <div style={{maxWidth:1400,margin:'0 auto',padding:'24px',display:'grid',gridTemplateColumns:'300px 1fr',gap:24}}>
        {/* LEFT */}
        <div style={{display:'flex',flexDirection:'column',gap:16}}>
          <div style={{background:'#fff',borderRadius:16,padding:20,boxShadow:'0 1px 8px #0001'}}>
            <div style={{fontWeight:700,fontSize:14,color:'#1e293b',marginBottom:12}}>📋 Select Task</div>
            <div style={{display:'flex',flexDirection:'column',gap:8}}>
              {Object.entries(taskMeta).map(([id,meta])=>(
                <TaskCard key={id} id={id} meta={meta} selected={taskName===id} onSelect={t=>{setTaskName(t);reset(t)}}/>
              ))}
            </div>
          </div>

          {obs&&<div style={{background:'#fff',borderRadius:16,padding:20,boxShadow:'0 1px 8px #0001'}}>
            <div style={{fontWeight:700,fontSize:14,color:'#1e293b',marginBottom:12}}>📊 Episode Stats</div>
            {[{label:'Steps Used',val:`${obs.step_number}/${obs.max_steps}`,color:'#6366f1'},
              {label:'Phases Done',val:`${completedPhases}/7`,color:'#10b981'},
              {label:'Cumul. Reward',val:totalReward.toFixed(3),color:'#f59e0b'},
              {label:'Best Raw Score',val:bestRaw.toFixed(3),color:'#ef4444'}].map(s=>(
              <div key={s.label} style={{display:'flex',justifyContent:'space-between',padding:'6px 0',borderBottom:'1px solid #f1f5f9'}}>
                <span style={{fontSize:13,color:'#64748b'}}>{s.label}</span>
                <span style={{fontWeight:700,color:s.color,fontSize:13}}>{s.val}</span>
              </div>
            ))}
          </div>}

          {rewardHistory.length>0&&<div style={{background:'#fff',borderRadius:16,padding:20,boxShadow:'0 1px 8px #0001'}}>
            <div style={{fontWeight:700,fontSize:14,color:'#1e293b',marginBottom:12}}>📈 Reward Curve</div>
            <ResponsiveContainer width="100%" height={140}>
              <LineChart data={rewardHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>
                <XAxis dataKey="step" tick={{fontSize:10}}/>
                <YAxis domain={[0,1]} tick={{fontSize:10}}/>
                <Tooltip formatter={v=>v.toFixed(3)}/>
                <Line type="monotone" dataKey="reward" stroke="#6366f1" strokeWidth={2} dot={{r:3}} name="Reward"/>
                <Line type="monotone" dataKey="raw" stroke="#f59e0b" strokeWidth={2} dot={{r:3}} name="Raw" strokeDasharray="5 5"/>
              </LineChart>
            </ResponsiveContainer>
          </div>}
        </div>

        {/* RIGHT */}
        <div style={{display:'flex',flexDirection:'column',gap:16}}>
          {obs&&<div style={{background:'#fff',borderRadius:16,padding:20,boxShadow:'0 1px 8px #0001'}}>
            <div style={{fontWeight:700,fontSize:14,color:'#1e293b',marginBottom:10}}>🔄 Research Workflow</div>
            <PhaseProgress progress={obs.progress} current={obs.current_phase}/>
          </div>}

          {obs&&<div style={{background:'#fff',borderRadius:16,padding:20,boxShadow:'0 1px 8px #0001'}}>
            <div style={{fontWeight:700,fontSize:14,color:'#1e293b',marginBottom:8}}>📄 Research Context</div>
            <pre style={{fontSize:12,color:'#475569',whiteSpace:'pre-wrap',lineHeight:1.7,margin:0,maxHeight:200,overflowY:'auto',background:'#f8fafc',borderRadius:8,padding:12}}>{obs.research_context}</pre>
            {obs.hints?.length>0&&obs.hints.map((h,i)=>(
              <div key={i} style={{fontSize:12,color:'#0369a1',background:'#f0f9ff',borderRadius:6,padding:'5px 10px',marginTop:6}}>💡 {h}</div>
            ))}
          </div>}

          {obs&&!done&&<div style={{background:'#fff',borderRadius:16,padding:20,boxShadow:'0 1px 8px #0001'}}>
            <div style={{fontWeight:700,fontSize:14,color:'#1e293b',marginBottom:12}}>🎯 Submit Action</div>
            <div style={{display:'flex',gap:8,flexWrap:'wrap',marginBottom:12}}>
              {(obs.allowed_actions||PHASE_ORDER).map(a=>(
                <button key={a} onClick={()=>setActionType(a)} style={{
                  padding:'6px 14px',borderRadius:20,border:`2px solid ${actionType===a?phaseColors[a]:'#e2e8f0'}`,
                  background:actionType===a?phaseColors[a]:'#fff',color:actionType===a?'#fff':'#64748b',
                  fontSize:12,fontWeight:600,cursor:'pointer',transition:'all 0.2s'}}>{PHASE_LABELS[a]||a}
                </button>
              ))}
            </div>
            <textarea value={actionContent} onChange={e=>setActionContent(e.target.value)}
              placeholder={`Write your ${PHASE_LABELS[actionType]||actionType} response here… Be detailed and technical.`}
              style={{width:'100%',minHeight:120,borderRadius:10,border:'2px solid #e2e8f0',padding:12,
                fontSize:13,color:'#1e293b',resize:'vertical',outline:'none',boxSizing:'border-box',fontFamily:'inherit',lineHeight:1.6}}
              onFocus={e=>e.target.style.borderColor='#6366f1'} onBlur={e=>e.target.style.borderColor='#e2e8f0'}/>
            <div style={{display:'flex',gap:10,marginTop:10,alignItems:'center'}}>
              <button onClick={step} disabled={loading||!actionContent.trim()} style={{
                padding:'10px 28px',borderRadius:10,border:'none',
                background:loading?'#e2e8f0':'linear-gradient(135deg,#6366f1,#8b5cf6)',
                color:loading?'#94a3b8':'#fff',fontWeight:700,fontSize:14,
                cursor:loading?'not-allowed':'pointer'}}>
                {loading?'⏳ Submitting…':'▶ Submit Action'}
              </button>
              <button onClick={()=>reset(taskName)} style={{padding:'10px 20px',borderRadius:10,border:'2px solid #e2e8f0',background:'#fff',color:'#64748b',fontWeight:600,fontSize:13,cursor:'pointer'}}>🔄 Reset</button>
              <span style={{color:'#94a3b8',fontSize:12,marginLeft:'auto'}}>{actionContent.length} chars</span>
            </div>
            {error&&<div style={{marginTop:8,color:'#ef4444',fontSize:12}}>⚠️ {error}</div>}
          </div>}

          {done&&<div style={{background:'linear-gradient(135deg,#10b981,#059669)',borderRadius:16,padding:20,color:'#fff',textAlign:'center'}}>
            <div style={{fontSize:32,marginBottom:8}}>🎉</div>
            <div style={{fontWeight:800,fontSize:20,marginBottom:4}}>Episode Complete!</div>
            <div style={{fontSize:15}}>Final Score: <strong>{bestRaw.toFixed(3)}</strong> | Total Reward: <strong>{totalReward.toFixed(3)}</strong></div>
            <button onClick={()=>reset(taskName)} style={{marginTop:14,padding:'10px 28px',borderRadius:10,border:'2px solid #fff',background:'transparent',color:'#fff',fontWeight:700,cursor:'pointer',fontSize:14}}>🔄 New Episode</button>
          </div>}

          {history.length>0&&<div style={{background:'#fff',borderRadius:16,padding:20,boxShadow:'0 1px 8px #0001'}}>
            <div style={{fontWeight:700,fontSize:14,color:'#1e293b',marginBottom:12}}>📜 Action History</div>
            <div style={{maxHeight:400,overflowY:'auto'}}>
              {[...history].reverse().map((h,i)=><HistoryItem key={i} entry={h} phaseColors={phaseColors}/>)}
            </div>
          </div>}

          {history.length>1&&<div style={{background:'#fff',borderRadius:16,padding:20,boxShadow:'0 1px 8px #0001'}}>
            <div style={{fontWeight:700,fontSize:14,color:'#1e293b',marginBottom:12}}>📊 Score by Phase</div>
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={history.map(h=>({name:(PHASE_LABELS[h.action_type]||h.action_type).split(' ')[0],score:+(h.info?.raw_score||0).toFixed(3),reward:+h.reward.toFixed(3)}))}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>
                <XAxis dataKey="name" tick={{fontSize:10}}/>
                <YAxis domain={[0,1]} tick={{fontSize:10}}/>
                <Tooltip/><Legend/>
                <Bar dataKey="score" fill="#6366f1" name="Raw Score" radius={[4,4,0,0]}/>
                <Bar dataKey="reward" fill="#10b981" name="Shaped Reward" radius={[4,4,0,0]}/>
              </BarChart>
            </ResponsiveContainer>
          </div>}
        </div>
      </div>
    </div>
  )
}
