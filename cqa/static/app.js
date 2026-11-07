// CQA reviewer UI (vanilla JS, no build step). API: docs/API.md, screens: docs/UI.md
const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const api=async(u,o={})=>{const r=await fetch(u,o);if(!r.ok){let t=await r.text();try{t=JSON.parse(t).detail||t}catch(e){}throw new Error(t)}return r.json()};
const J=b=>({method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let APS=[],APOPEN=null,APSTAT='open',G=null,RUN='',DOC=null,DIALOGS={},CUR=null,SORT={k:'total',d:1},FILT='all',CL=null,CLS={},ORDER=[],OPEN=null;
const SPK={client:'Клиент',operator:'Оператор',bot:'Бот',system:'Система'};
const CHN=c=>/call|voice/.test(c||'')?'Звонок':'Чат';
const MODEL=m=>String(m||'').replace(/^\d+:/,'').replace('claude-','').replace(/-5-5$/,'').replace(/^./,x=>x.toUpperCase());
function toast(m){$$('.toast').forEach(x=>x.remove());const t=document.createElement('div');t.className='toast';t.textContent=m;document.body.appendChild(t);setTimeout(()=>t.remove(),2600)}
function cdef(clid,cid){return CLS[clid]?.criteria.find(x=>x.id==cid)}
// semantic class: good for the operator = green, bad = red (gross criteria: YES = bad)
function cls(v,def){if(v===null||v===undefined)return'err';if(v==='NA')return'na';
 if(v==='YES'||v==='NO'){const good=def&&def.gross?v==='NO':v==='YES';return good?'good':'bad'}
 const sc=(def&&def.scale)||[];const mx=sc.length?Math.max(...sc):20;const n=+v;return n>=mx?'good':n<=0?'bad':'mid'}
const VL={YES:'Да',NO:'Нет',NA:'н/п'};
function badge(v,def,extra=''){return `<span class="v ${cls(v,def)} ${extra}">${v===null||v===undefined?'ошибка':VL[v]??v}</span>`}
function judgeTip(c){if(!c.jury)return'';const j=Object.entries(c.jury).map(([m,v])=>`${MODEL(m)}: ${VL[v]??v}`).join('<br>');
 return c.decided_by&&c.decided_by!=='consensus'?`Присяжные разошлись:<br>${j}<br><b>Судья ${MODEL(c.decided_by)}: ${VL[c.value]??c.value}</b>`:`Единогласно:<br>${j}`}
document.addEventListener('mouseover',e=>{const t=e.target.closest('[data-tip]');const tp=$('#tip');if(!t||!t.dataset.tip){tp.classList.add('hide');return}tp.innerHTML=t.dataset.tip;tp.classList.remove('hide');const r=t.getBoundingClientRect();tp.style.left=Math.max(8,Math.min(innerWidth-330,r.left))+'px';tp.style.top=(r.bottom+6)+'px'});
function view(v){$$('#tabs a').forEach(a=>a.classList.toggle('on',a.dataset.v==v));['runs','dialog','appeals','checklists'].forEach(x=>$('#v-'+x).classList.toggle('hide',x!=v));$('#tip').classList.add('hide');if(v=='checklists')loadCl();if(v=='appeals')loadAppeals();window.scrollTo(0,0)}
$$('#tabs a').forEach(a=>a.onclick=()=>view(a.dataset.v));
const curView=()=>$$('#tabs a.on')[0].dataset.v;
$('#btnMore').onclick=e=>{e.stopPropagation();$('#pop').classList.toggle('hide')};
document.addEventListener('click',e=>{if(!e.target.closest('.menu'))$('#pop').classList.add('hide')});
async function init(){await loadAllCl();const gs=await api('/api/groups');
 $('#gsel').innerHTML=gs.map(g=>`<option value="${esc(g.group)}">${esc(g.group)} · ${g.dialogs}</option>`).join('');
 const want=localStorage.cqaGroup&&gs.find(g=>g.group==localStorage.cqaGroup)?localStorage.cqaGroup:(gs.find(g=>g.group=='DEV')||gs[0]||{}).group;
 if(want){$('#gsel').value=want;await loadGroup(want)}else $('#ovw').innerHTML='<div class="card empty">Нет данных. Загрузите группу или JSON-файл.</div>'}
$('#gsel').onchange=()=>loadGroup($('#gsel').value);
$('#rsel').onchange=()=>{RUN=$('#rsel').value;loadResults()};
async function loadGroup(g,keepRun){G=g;localStorage.cqaGroup=g;
 try{const ds=await api('/api/dialogs/'+encodeURIComponent(g));DIALOGS={};ds.forEach(d=>DIALOGS[d.conversation_id]=d)}catch(e){DIALOGS={}}
 let runs=[];try{runs=await api('/api/runs/'+encodeURIComponent(g))}catch(e){}
 if(!keepRun||!runs.find(r=>r.run==RUN))RUN=runs[0]?.run||'';
 $('#rsel').innerHTML=runs.length?runs.map((r,i)=>`<option value="${esc(r.run)}">${esc(r.run)} · ${r.dialogs} · ${esc(r.created_at.slice(5,16).replace('T',' '))}${i==0?' · новый':''}</option>`).join(''):'<option value="">нет результатов</option>';
 $('#rsel').value=RUN;await refreshAps();await loadResults()}
const rq=()=>RUN?'?run='+encodeURIComponent(RUN):'';
async function loadResults(){try{DOC=await api('/api/results/'+encodeURIComponent(G)+rq())}catch(e){DOC=null}
 $('#exJson').href=`/api/export/${G}.json${rq()}`;$('#exCsv').href=`/api/export/${G}.csv?which=both${RUN?'&run='+encodeURIComponent(RUN):''}`;$('#exV1').href=`/api/export/${G}/results.json${rq()}`;$('#exOrg').href=`/api/export/${G}.csv?which=organizer${RUN?'&run='+encodeURIComponent(RUN):''}`;
 renderOverview();renderTable();if(CUR&&curView()=='dialog')renderDialog()}
// ---------- overview ----------
function stats(){const ds=DOC.dialogs;const tot=d=>d.human&&d.human.reviewed?d.human.total:d.total;
 const t=ds.map(tot).filter(x=>x!=null);const per={};let judge=0,cells=0;
 ds.forEach(d=>d.criteria.forEach(c=>{const def=cdef(d.checklist_id,c.criterion_id);const p=per[c.criterion_id]=per[c.criterion_id]||{name:c.name,def,n:0,b:{g:0,g2:0,m:0,b:0,n:0},fail:[]};
  const hv=d.human?.overrides?.[c.criterion_id];const v=hv?hv.value:c.value;p.n++;cells++;if(c.decided_by&&c.decided_by!=='consensus')judge++;
  const k=cls(v,def);if(k=='na')p.b.n++;else if(k=='good')p.b.g++;else if(k=='mid'){const mx=Math.max(...((def?.scale)||[20]));if(+v>=mx/2)p.b.g2++;else p.b.m++;p.fail.push(d.conversation_id)}else{p.b.b++;p.fail.push(d.conversation_id)}}));
 return{ds,t,per,judge,cells,gross:ds.filter(d=>(d.human?.reviewed?d.human.gross_error:d.gross_error)),rev:ds.filter(d=>d.human&&d.human.reviewed)}}
function renderOverview(){if(!DOC){$('#ovw').innerHTML=`<div class="card empty" style="margin-bottom:16px">Группа <b>${esc(G)}</b> ещё не оценена — нажмите «Загрузить и оценить» или «Ещё → Переоценить».</div>`;return}
 const S=stats(),m=DOC.meta||{},avg=S.t.length?S.t.reduce((a,b)=>a+b,0)/S.t.length:null;
 const bins=Array(10).fill(0);S.t.forEach(x=>bins[Math.min(9,Math.floor(x/10))]++);const mx=Math.max(...bins,1);
 const srt=S.t.slice().sort((a,b)=>a-b),med=srt.length?srt[Math.floor(srt.length/2)]:null;
 const fails=Object.entries(S.per).filter(([k,p])=>!p.def?.gross).map(([k,p])=>({k,p,r:p.fail.length/Math.max(1,p.n-p.b.n)})).sort((a,b)=>b.r-a.r);
 const ins=[];const f0=fails[0],f1=fails[1];const chip=d=>`<span class="chip" onclick="openDialog('${esc(d.conversation_id)}')">${esc(d.conversation_id)}</span>`;
 if(f0&&f0.p.fail.length)ins.push(`<li><span class="dot r"></span><div>Чаще всего не выполняется <b>${f0.k} «${esc(f0.p.name)}»</b> — ${Math.round(f0.r*100)}% применимых диалогов (${f0.p.fail.length}).${f1&&f1.p.fail.length?` Далее — ${f1.k} «${esc(f1.p.name)}», ${Math.round(f1.r*100)}%.`:''}</div></li>`);
 ins.push(S.gross.length?`<li><span class="dot r"></span><div><b>${S.gross.length}</b> с грубой ошибкой:<br>${S.gross.map(chip).join('')}</div></li>`:`<li><span class="dot g"></span><div>Грубых ошибок не обнаружено.</div></li>`);
 const inc=S.ds.filter(d=>d.status=='incomplete');if(inc.length)ins.unshift(`<li><span class="dot r"></span><div><b>${inc.length}</b> неполных оценок (ошибка модели по критерию — итог не выставлен, нужна переоценка или ручная оценка):<br>${inc.map(chip).join('')}</div></li>`);
 const rv=S.ds.filter(d=>d.human?.reviewed);if(rv.length){let n=0,same=0;const per={};rv.forEach(d=>{const ov=d.human.overrides||{};Object.keys(ov).forEach(cid=>{if(ov[cid]?.value==null)return;n++;const c=d.criteria.find(x=>x.criterion_id==cid);const ok=String(c?.value)==String(ov[cid].value);same+=ok;(per[cid]=per[cid]||[0,0])[0]+=ok;per[cid][1]++})});
  if(n)ins.push(`<li><span class="dot ${same/n>=.8?'g':'r'}"></span><div>Согласие AI↔человек на проверенных критериях: <b>${Math.round(100*same/n)}%</b> (${same}/${n}; ${rv.length} диалогов)${Object.entries(per).sort().map(([k,[a,b]])=>` · ${k}: ${a}/${b}`).join('')}</div></li>`)}
 const nr=S.ds.filter(d=>(d.needs_review||[]).length),nrc=S.ds.reduce((a,d)=>a+(d.needs_review||[]).length,0);if(nr.length)ins.push(`<li><span class="dot r"></span><div>Требуют проверки человеком: <b>${nrc}</b> критериев в <b>${nr.length}</b> диалогах (присяжные разошлись / решал судья / шкала с низкой уверенностью): ${nr.slice(0,12).map(chip).join('')}${nr.length>12?' …':''}</div></li>`);
 const low=S.ds.filter(d=>d.total!=null&&d.total<50&&!d.gross_error);if(low.length)ins.push(`<li><span class="dot r"></span><div>Итог ниже 50 (без грубых): ${low.map(chip).join('')}</div></li>`);
 const jm=MODEL((m.model||'').match(/judge=([\w.-]+)/)?.[1]||'');
 ins.push(`<li><span class="dot"></span><div>Присяжные разошлись в <b>${S.judge}</b> из ${S.cells} оценок (${(100*S.judge/Math.max(1,S.cells)).toFixed(0)}%) — решение принял судья${jm?' '+jm:''}.</div></li>`);
 ins.push(`<li><span class="dot p"></span><div>Проверено человеком: <b>${S.rev.length}</b> из ${S.ds.length}.${S.rev.length?'':' Откройте диалог, чтобы подтвердить или исправить оценку.'}</div></li>`);
 const jury=(m.model||'').replace(/^jury\[|\].*$/g,'').split('+').map(MODEL).join(' + ');
 $('#ovw').innerHTML=`<div class="ov">
 <div class="card hero"><div class="lbl">Средний балл качества</div><div class="big num">${avg==null?'—':avg.toFixed(1)}<small> / 100</small></div>
  <div class="mut sm" style="margin-top:6px">медиана ${med==null?'—':med.toFixed(1)} · группа ${esc(G)}</div>
  <div class="kpis"><div class="kpi"><b class="num">${S.ds.length}</b><span>диалогов</span></div><div class="kpi ${S.gross.length?'bad':''}"><b class="num">${S.gross.length}</b><span>грубых ошибок</span></div>
  <div class="kpi"><b class="num">${S.judge}</b><span>оценок решил судья</span></div><div class="kpi ${S.ds.some(d=>(d.needs_review||[]).length)?'bad':''}" id="kpi-review"><b class="num">${S.ds.reduce((a,d)=>a+(d.needs_review||[]).length,0)}</b><span>требуют проверки</span></div><div class="kpi"><b class="num">${S.rev.length}</b><span>проверено человеком</span></div></div></div>
 <div class="card"><h3>Распределение итоговых баллов</h3><div class="hist">${bins.map((b,i)=>`<div class="col ${i<5?'lo':i<7?'mid':''} ${b?'':'zero'}" data-tip="Итог ${i*10}–${i*10+10}: ${b} диал."><em>${b}</em><i style="height:${b/mx*112}px"></i></div>`).join('')}</div>
  <div class="hax">${bins.map((b,i)=>`<span>${i*10}–${i*10+10}</span>`).join('')}</div><div class="mut sm" style="margin-top:10px">Число диалогов по диапазонам итогового балла (с учётом ручных правок). Грубая ошибка помечается отдельно и в балл не входит.</div></div>
 <div class="card insc"><h3>Выводы</h3><ul class="ins">${ins.join('')}</ul></div></div>
 <div class="card crits-ov"><div style="display:flex;align-items:center;margin-bottom:8px"><h3 style="margin:0">Выполнение критериев</h3><span class="sp"></span>
  <div class="legend"><span><i style="background:#34a853"></i>Выполнен</span><span><i style="background:#86cf98"></i>Частично ≥ ½</span><span><i style="background:#ff9f0a"></i>Частично &lt; ½</span><span><i style="background:#e5484d"></i>Не выполнен / грубая</span><span><i style="background:#d1d1d6"></i>Неприменимо</span></div></div>
  ${Object.entries(S.per).map(([k,p])=>{const gr=p.def?.gross,sc=p.def?.type=='scale';const seg=[['g',p.b.g,gr?'Нет грубой ошибки':sc?'Максимум':'Да'],['g2',p.b.g2,'Частично ≥ ½'],['m',p.b.m,'Частично < ½'],['b',p.b.b,gr?'Грубая ошибка':sc?'0 баллов':'Нет'],['n',p.b.n,'Неприменимо']];
   const ap=p.n-p.b.n;return `<div class="crow"><div class="nm" title="${esc(p.name)}"><b>${k}</b>${esc(p.name)}</div>
   <div class="sbar">${seg.filter(s=>s[1]).map(s=>`<div class="${s[0]}" style="width:${100*s[1]/p.n}%" data-tip="${s[2]}: ${s[1]} из ${p.n}">${s[1]/p.n>.05?s[1]:''}</div>`).join('')}</div>
   <div class="mut sm num">${ap?(gr?`${p.b.b} случ. · ${Math.round(100*p.b.b/ap)}%`:`${Math.round(100*p.b.g/ap)}% выполнено`):'—'}${p.b.n?` · ${p.b.n} н/п`:''}</div></div>`}).join('')}
  <div class="mut sm" style="margin-top:12px;padding-top:10px;border-top:1px solid var(--line)">Прогон ${esc(DOC.run||RUN)} от ${esc((DOC.created_at||'').replace('T',' '))} · присяжные: ${esc(jury)}${jm?' · судья: '+esc(jm):''}${m.cost_usd!=null?' · стоимость $'+m.cost_usd:''}</div></div>`}
// ---------- table ----------
$$('#fseg a').forEach(a=>a.onclick=()=>{FILT=a.dataset.f;$$('#fseg a').forEach(x=>x.classList.toggle('on',x==a));renderTable()});
$('#flt').oninput=renderTable;
function critSort(d,cid){const c=d.criteria.find(x=>x.criterion_id==cid);if(!c)return -9;const o=d.human?.overrides?.[cid];return{err:-1,bad:0,mid:1,good:2,na:3}[cls(o?o.value:c.value,cdef(d.checklist_id,cid))]}
function rowsFiltered(){const f=$('#flt').value.toLowerCase().trim();let rows=DOC?DOC.dialogs.slice():Object.values(DIALOGS).map(d=>({conversation_id:d.conversation_id,language:d.language,channel:d.channel,criteria:[]}));
 rows=rows.filter(d=>!f||(d.conversation_id+' '+d.language+' '+d.channel+' '+CHN(d.channel)).toLowerCase().includes(f));
 if(FILT=='gross')rows=rows.filter(d=>d.human?.reviewed?d.human.gross_error:d.gross_error);if(FILT=='judge')rows=rows.filter(d=>d.criteria.some(c=>c.decided_by&&c.decided_by!=='consensus'));
 if(FILT=='low')rows=rows.filter(d=>d.total!=null&&d.total<70);if(FILT=='hum')rows=rows.filter(d=>d.human?.reviewed);
 const k=SORT.k,gv=d=>k=='total'?(d.total??-1):k=='human'?(d.human?.reviewed?d.human.total:-1):k=='gross'?(d.gross_error?1:0):k.startsWith('c:')?critSort(d,k.slice(2)):String(d[k]??'');
 rows.sort((a,b)=>{const x=gv(a),y=gv(b);return(x>y?1:x<y?-1:a.conversation_id>b.conversation_id?1:-1)*SORT.d});return rows}
function renderTable(){const rows=rowsFiltered();ORDER=rows.map(r=>r.conversation_id);
 $('#cnt').textContent=`${rows.length} из ${DOC?DOC.dialogs.length:Object.keys(DIALOGS).length}`;
 const all=DOC?DOC.dialogs:[];const cids=[],cname={};all.forEach(d=>d.criteria.forEach(c=>{if(!cids.includes(c.criterion_id)){cids.push(c.criterion_id);cname[c.criterion_id]=c.name}}));
 const anyH=all.some(d=>d.human?.reviewed);
 const th=(k,l,c='',tip='')=>`<th class="${c}" data-k="${k}" ${tip?`data-tip="${esc(tip)}"`:''}>${l}${SORT.k==k?`<span class="ar"> ${SORT.d>0?'↑':'↓'}</span>`:''}</th>`;
 if(!rows.length){$('#tbl').innerHTML=`<tbody><tr><td class="empty">Нет диалогов по выбранному фильтру.</td></tr></tbody>`;return}
 $('#tbl').innerHTML=`<thead><tr>${th('conversation_id','Диалог','l')}${th('channel','Канал','l')}${th('language','Язык')}${cids.map(c=>th('c:'+c,c,'',c+' · '+esc(cname[c]))).join('')}${th('total','Итог AI')}${anyH?th('human','После правок'):''}${th('gross','Грубая')}</tr></thead><tbody>`+
 rows.map(d=>{const g=d.human?.reviewed?d.human.gross_error:d.gross_error,tot=d.total;
  return `<tr class="${g?'gross':''}" onclick="openDialog('${esc(d.conversation_id)}')"><td class="l" style="font-family:ui-monospace,Menlo,monospace;font-size:12px">${esc(d.conversation_id)}</td><td class="l mut">${CHN(d.channel)}</td><td class="mut">${esc(d.language||'')}</td>
  ${cids.map(cid=>{const c=d.criteria.find(x=>x.criterion_id==cid);if(!c)return'<td></td>';const def=cdef(d.checklist_id,cid),o=d.human?.overrides?.[cid],jd=c.decided_by&&c.decided_by!=='consensus';
   const tip=`<b>${cid} · ${esc(c.name)}</b><br>`+judgeTip(c)+(o?`<br><span style="color:#c9b3ff">Человек: ${VL[o.value]??o.value}</span>`:'');
   return `<td data-tip="${esc(tip)}">${badge(o?o.value:c.value,def,o?'hum':'')}${jd?'<span class="jd">С</span>':''}</td>`}).join('')}
  <td>${d.status=='incomplete'?`<span class="tag bad" data-tip="Не оценены: ${esc((d.failed_criteria||[]).join(', '))}${d.provisional_total!=null?' · предварительно '+Math.round(d.provisional_total):''}">неполная</span>`:`<span class="tot num ${tot!=null&&tot<50?'lo':''}">${tot==null?'—':Math.round(tot)}</span>`}</td>${anyH?`<td>${d.human?.reviewed?`<span class="tot num" style="color:var(--hum)">${Math.round(d.human.total)}</span>`:''}</td>`:''}
  <td>${g?'<span class="tag bad">Да</span>':''}</td></tr>`}).join('')+'</tbody>';
 $$('#tbl th[data-k]').forEach(th=>th.onclick=()=>{SORT={k:th.dataset.k,d:SORT.k==th.dataset.k?-SORT.d:1};renderTable()})}
// ---------- dialog ----------
function fmt(s){if(s==null)return'';s=Math.floor(s);return String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0')}
function openDialog(id){CUR=id;OPEN=null;if(!ORDER.includes(id))ORDER=(DOC?DOC.dialogs:[]).map(d=>d.conversation_id);$('#tabs a[data-v=dialog]').classList.remove('dis');view('dialog');renderDialog()}
function renderDialog(){const rec=DIALOGS[CUR],res=DOC&&DOC.dialogs.find(d=>d.conversation_id==CUR);
 if(!rec){$('#dhead').innerHTML='<span class="mut">Нет данных диалога</span>';return}
 const ev={};(res?res.criteria:[]).forEach(c=>(c.evidence||[]).forEach(e=>{(ev[e.turn]=ev[e.turn]||[]).includes(c.criterion_id)||ev[e.turn].push(c.criterion_id)}));
 const h=res&&res.human,pos=ORDER.indexOf(CUR),g=res&&(h?.reviewed?h.gross_error:res.gross_error);
 $('#dhead').innerHTML=`<button class="btn" onclick="view('runs')">← Обзор</button><span class="id">${esc(rec.conversation_id)}</span>
  <span class="tag soft">${CHN(rec.channel)}</span><span class="tag soft">${esc(rec.language)}</span>${g?'<span class="tag bad">Грубая ошибка</span>':''}${h?.reviewed?'<span class="tag hum">Проверено человеком</span>':''}${res&&res.status=='incomplete'?`<span class="tag bad" id="incTag" data-tip="Итог не выставлен: оценка критериев ${esc((res.failed_criteria||[]).join(', '))} завершилась ошибкой">Неполная оценка: ${esc((res.failed_criteria||[]).join(', '))}</span>`:''}${res&&(res.warnings||[]).length?`<span class="tag soft" id="warnTag" data-tip="${esc(res.warnings.join(' | '))}">⚠ ${res.warnings.length}</span>`:''}
  <span class="sp"></span>
  <div class="score"><b class="num">${res&&res.total!=null?res.total.toFixed(1):'—'}</b><span>/ 100 · итог AI${res&&res.provisional_total!=null?` (предварительно ${res.provisional_total.toFixed(1)})`:''}</span></div>${res&&res.stability&&res.stability.score_range&&res.stability.score_range[0]!=null?`<div class="meta" id="stability" data-tip="Разброс итога, если заменить итоговое значение критерия любым голосом присяжных/выборок судьи">Стабильность: согласие ${res.stability.criteria_unanimous}/${res.stability.criteria_total} критериев · диапазон итога ${res.stability.score_range[0].toFixed(1)}–${res.stability.score_range[1].toFixed(1)} · требуют проверки ${res.stability.flagged_criteria}</div>`:''}${res&&res.call_metrics?`<div class="meta" id="callmetrics" data-tip="Информационно, по меткам времени без звука; в оценку не входит">Метрики звонка: темп оператора ${res.call_metrics.operator_wpm??'—'} слов/мин · перебивания ${res.call_metrics.overlap_rate==null?'—':Math.round(100*res.call_metrics.overlap_rate)+'%'} · пауза до ответа оператора: медиана ${res.call_metrics.operator_response_s.median??'—'} с, p90 ${res.call_metrics.operator_response_s.p90??'—'} с, макс ${res.call_metrics.operator_response_s.max??'—'} с</div>`:''}
  ${h?.reviewed?`<div class="score h"><b class="num">${h.total!=null?h.total.toFixed(1):'—'}</b><span>/ 100 · после правок</span></div>`:''}
  <span class="mut sm num">${pos>=0?`диалог ${pos+1} из ${ORDER.length}`:''}</span>
  <button class="btn" onclick="nav(-1)" ${pos<=0?'disabled':''} title="Предыдущий (k)">↑</button><button class="btn" onclick="nav(1)" ${pos<0||pos>=ORDER.length-1?'disabled':''} title="Следующий (j)">↓</button>
  <button class="btn" onclick="reevalOne()" title="Повторная оценка только этого диалога: модели уходят только изменённые критерии (кэш; «Без кэша» в меню «Ещё» — всё заново)">Переоценить</button>`;
 const isCall=rec.kind=='call';
 $('#tmeta').textContent=`${rec.transcript.length} реплик (нумерация с №1)${isCall?' · клик по реплике — перемотка записи':''}`;
 $('#audioBox').innerHTML=isCall&&rec.audio_url?`<div class="audio"><audio id="aud" controls preload="none" src="${esc(rec.audio_url)}"></audio><div class="mut sm">Перемотка приблизительная: смещение отсчитывается от первой реплики.</div></div>`:'';
 $('#transcript').innerHTML=rec.transcript.map(t=>`<div id="t${t.index}" class="turn ${esc(t.speaker)} ${t.internal?'internal':''} ${ev[t.index]?'ev':''} ${isCall?'seek':''}" ${isCall?`onclick="seek(${t.offset??0})"`:''}>
  <div class="n">${t.index}</div><div class="bub"><div class="who">${SPK[t.speaker]||esc(t.speaker)}${t.internal?' · внутреннее':''}<time>${fmt(t.offset)}</time>${(ev[t.index]||[]).map(c=>`<span class="evm" data-tip="Цитата для критерия ${c}">${c}</span>`).join('')}</div><div class="tx">${esc(t.text)}</div></div></div>`).join('');
 if(!res){$('#crits').innerHTML='<div class="empty">Диалог не оценён.</div>';$('#cmeta').textContent='';return}
 const jn=res.criteria.filter(c=>c.decided_by&&c.decided_by!=='consensus').length;
 $('#cmeta').textContent=`${res.criteria.length} критериев${jn?` · судья решил ${jn}`:' · модели единогласны'}${h&&Object.keys(h.overrides||{}).length?` · правок: ${Object.keys(h.overrides).length}`:''}`;
 $('#crits').innerHTML=res.criteria.map(c=>{const def=cdef(res.checklist_id,c.criterion_id),o=h?.overrides?.[c.criterion_id],cid=c.criterion_id;
  const opts=(c.type=='scale'?(def?.scale||[]).map(String):['YES','NO']).concat(def?.allow_na?['NA']:[]),jd=c.decided_by&&c.decided_by!=='consensus';
  return `<div class="cr ${o?'sel':''}" id="cr_${cid}"><div class="hd"><span class="t"><b>${cid}</b>${esc(c.name)}${def?.gross?' <span class="tag bad">грубая</span>':''}</span>
   ${o?`<span data-tip="Оценка AI">${badge(c.value,def)}</span><span class="mut">→</span><span data-tip="Оценка человека">${badge(o.value,def,'hum')}</span>`:badge(c.value,def)}${c.type=='scale'&&def?.scale?`<span class="mut sm">из ${Math.max(...def.scale)}</span>`:''}</div>
   <div class="js">${esc(c.justification)}</div>
   ${c.error?`<div class="meta" style="color:var(--bad)">Ошибка оценки: ${esc(c.error)} — критерий не засчитан как NA; выставьте оценку вручную или переоцените.</div>`:''}${c.needs_review?`<div class="meta" style="color:var(--bad)">Требует проверки: ${esc(((c.review_reasons&&c.review_reasons.length)?c.review_reasons:['gross_without_quote']).map(r=>({jurors_disagreed:'присяжные разошлись',judge_decided:'решение принял судья',judge_samples_split:'выборки судьи разошлись',low_confidence_scale:'шкала, низкая уверенность',gross_without_quote:'грубая ошибка без подтверждённой цитаты оператора'})[r]||r).join(', '))}${c.judge_samples&&c.judge_samples.length>1?` · выборки судьи: ${esc(c.judge_samples.join(' / '))} → медиана`:''}</div>`:''}${c.evidence_warning&&!c.needs_review?`<div class="meta">⚠ Доказательства: ${esc(c.evidence_warning)}</div>`:''}
   ${(c.client_requests||[]).length?`<div class="meta">Запросы клиента: ${c.client_requests.map(r=>`${r.status=='closed'?'✓':r.status=='partial'?'◐':'✗'} ${esc(r.text||'')}${r.essential?'':' <span class="mut">(второстеп.)</span>'}`).join(' · ')}</div>`:''}
   ${c.na_reason?`<div class="meta">Неприменимо: ${esc(c.na_reason)}</div>`:''}${c.missing_where?`<div class="meta">Ожидалось: ${esc(c.missing_where)}</div>`:''}
   ${(c.evidence||[]).length?`<div class="evs">${c.evidence.slice().sort((a,b)=>a.turn-b.turn).map(e=>`<div class="evq" onclick="focusTurn(${e.turn})" title="Показать в транскрипте"><b>#${e.turn}</b>«${esc(e.quote)}»</div>`).join('')}</div>`:''}
   <div class="jury">${c.jury?Object.entries(c.jury).map(([m,v])=>`<span class="jv">${MODEL(m)} ${badge(v,def)}</span>`).join(''):''}
    ${jd?`<span class="jv dec" data-tip="Присяжные разошлись — решение принял судья">Судья ${MODEL(c.decided_by)} → ${VL[c.value]??c.value}</span>`:c.jury?'<span class="mut">единогласно</span>':''}
    ${c.confidence!=null?`<span class="mut">· уверенность ${Math.round(c.confidence*100)}%</span>`:''}</div>
   ${o?`<div class="hn">Правка${o.reviewer?' ('+esc(o.reviewer)+')':''}: ${VL[c.value]??c.value} → <b>${VL[o.value]??o.value}</b>${o.justification?` — ${esc(o.justification)}`:''}</div>`:''}
   ${OPEN==cid?`<div class="ovf"><select id="ov_${cid}"><option value="AI">Как у AI (${VL[c.value]??c.value})</option>${opts.map(x=>`<option value="${x}" ${o&&String(o.value)==x?'selected':''}>${VL[x]??x}</option>`).join('')}</select>
    <input id="oj_${cid}" placeholder="Комментарий ревьюера" value="${esc(o?o.justification:'')}"><input id="or_${cid}" placeholder="Имя" style="width:110px;flex:0 0 110px;min-width:0" value="${esc(o?o.reviewer:localStorage.reviewer||'')}">
    <button class="btn pri" onclick="saveOv('${cid}')">Сохранить</button><button class="btn" onclick="OPEN=null;renderDialog()">Отмена</button></div>`:
    `${apBlock(cid,c)}<div class="ovr"><span class="lnk" onclick="openOv('${cid}')">${o?'Изменить правку':'Исправить оценку'}</span>${apFor(CUR,cid)?'':`<span class="lnk" id="apBtn_${cid}" onclick="APOPEN='${cid}';renderDialog();$('#apc_${cid}').focus()">Оспорить</span>`}${o?`<span class="lnk" style="color:var(--mut)" onclick="clearOv('${cid}')">Вернуть оценку AI</span>`:''}${c.reasoning?`<details style="display:inline"><summary style="margin:0">Рассуждение модели</summary><div class="rs">${esc(c.reasoning)}</div></details>`:''}</div>`}</div>`}).join('')}
function openOv(cid){const st=$('#crits').scrollTop;OPEN=cid;renderDialog();$('#crits').scrollTop=st;$('#ov_'+cid).focus()}
function nav(d){const i=ORDER.indexOf(CUR)+d;if(i>=0&&i<ORDER.length){CUR=ORDER[i];OPEN=null;renderDialog();$('#transcript').scrollTop=0;$('#crits').scrollTop=0}}
document.addEventListener('keydown',e=>{if(curView()!='dialog'||e.target.matches('input,textarea,select')||e.metaKey||e.ctrlKey||e.altKey)return;if(e.key=='j')nav(1);if(e.key=='k')nav(-1);if(e.key=='Escape')view('runs')});
function focusTurn(i){const el=$('#t'+i);if(!el)return;$$('.turn.flash').forEach(x=>x.classList.remove('flash'));el.classList.add('flash');el.scrollIntoView({block:'center',behavior:'smooth'});setTimeout(()=>el.classList.remove('flash'),2500);
 const t=DIALOGS[CUR].transcript.find(x=>x.index==i);if(DIALOGS[CUR].kind=='call'&&t)seek(t.offset||0,false)}
function seek(s,play=true){const a=$('#aud');if(a){try{a.currentTime=s}catch(e){}if(play)a.play().catch(()=>{})}}
async function putOv(cid,body){const st=$('#crits').scrollTop;const d=await api(`/api/overrides/${encodeURIComponent(G)}/${encodeURIComponent(CUR)}/${encodeURIComponent(cid)}${rq()}`,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 const i=DOC.dialogs.findIndex(x=>x.conversation_id==CUR);if(i>=0)DOC.dialogs[i]=d;OPEN=null;renderDialog();$('#crits').scrollTop=st;renderTable();renderOverview();return d}
async function saveOv(cid){const v=$('#ov_'+cid).value;localStorage.reviewer=$('#or_'+cid).value;
 try{const d=await putOv(cid,{value:v=='AI'?null:v,justification:$('#oj_'+cid).value,reviewer:$('#or_'+cid).value});toast(v=='AI'?'Оценка AI восстановлена':`Сохранено · итог после правок ${d.human?.total!=null?d.human.total.toFixed(1):'—'}`)}catch(e){toast('Ошибка: '+e.message)}}
async function clearOv(cid){try{await putOv(cid,{value:null});toast('Оценка AI восстановлена')}catch(e){toast('Ошибка: '+e.message)}}
// ---------- jobs ----------
const ST={queued:'В очереди',fetching:'Загрузка группы',running:'Оценка',done:'Готово',error:'Ошибка'};
async function runJob(body){const P=$('#prog');P.className='prog';$('#pclose').classList.add('hide');$('#pstat').textContent='Запуск…';$('#pbar').style.width='0';$('#pmeta').textContent='';
 let j;try{j=await api('/api/evaluate',J(body))}catch(e){P.classList.add('err');$('#pstat').textContent='Ошибка: '+e.message;$('#pclose').classList.remove('hide');return}
 const tick=async()=>{let s;try{s=await api('/api/jobs/'+j.id)}catch(e){setTimeout(tick,2000);return}$('#pbar').style.width=(s.status=='done'?100:s.total?100*s.done/s.total:0)+'%';
  $('#pstat').innerHTML=`<b>${ST[s.status]||esc(s.status)}</b> · группа ${esc(body.group)}${s.total?` · ${s.done} из ${s.total} оценок`:''}`;
  $('#pmeta').textContent=`${s.dialogs??'…'} диал. · ${s.elapsed_s} с${s.meta?' · $'+s.meta.cost_usd:''}${s.error?' · '+s.error:''}`;
  if(s.status=='done'||s.status=='error'){P.classList.add(s.status=='done'?'ok':'err');$('#pclose').classList.remove('hide');await reloadGroups(body.group);return}setTimeout(tick,1200)};tick()}
$('#pclose').onclick=()=>$('#prog').classList.add('hide');
async function reloadGroups(g){const gs=await api('/api/groups');$('#gsel').innerHTML=gs.map(x=>`<option value="${esc(x.group)}">${esc(x.group)} · ${x.dialogs}</option>`).join('');$('#gsel').value=g;await loadGroup(g)}
const fresh=()=>$('#fresh').checked;
// in-page dialog instead of window.prompt/confirm: ask(title,label,default) -> string|null; label=null -> yes/no
function ask(title,label,def=''){return new Promise(res=>{const m=$('#askM'),inp=$('#askI');$('#askT').textContent=title;
 $('#askL').textContent=label||'';inp.style.display=label==null?'none':'';inp.value=def;m.classList.remove('hide');if(label!=null)inp.focus();
 const done=v=>{m.classList.add('hide');$('#askOk').onclick=$('#askNo').onclick=inp.onkeydown=null;res(v)};
 $('#askOk').onclick=()=>done(label==null?'yes':inp.value.trim()||null);$('#askNo').onclick=()=>done(null);
 inp.onkeydown=e=>{if(e.key=='Enter')$('#askOk').click();if(e.key=='Escape')$('#askNo').click()}})}
$('#btnMain').onclick=async()=>{const g=await ask('Загрузить и оценить','ID группы в CQA API (например, DEV или ID контрольного набора)',G||'');if(g)runJob({group:g,fetch:true,fresh:fresh()})};
$('#mReeval').onclick=async()=>{$('#pop').classList.add('hide');if(await ask(`Переоценить группу ${G}${fresh()?' без кэша (все вызовы заново)':''}?`,null))runJob({group:G,fresh:fresh()})};
async function reevalOne(){if(await ask(`Переоценить диалог ${CUR} по текущему чек-листу?${fresh()?' Без кэша: все критерии заново.':' Модели уйдут только изменённые критерии.'}`,null))runJob({group:G,ids:[CUR],fresh:fresh()})}
$('#mFetch').onclick=async()=>{$('#pop').classList.add('hide');const g=await ask('Загрузить группу','ID группы в CQA API',G||'');if(!g)return;try{const r=await api('/api/fetch',J({group_id:g.trim()}));toast(`Загружено ${r.dialogs} диалогов`);await reloadGroups(r.group)}catch(e){toast('Ошибка: '+e.message)}};
$('#mFile').onclick=()=>{$('#pop').classList.add('hide');$('#file').click()};
$('#file').onchange=async e=>{const f=e.target.files[0];if(!f)return;const g=await ask('Загрузка JSON','Имя группы (латиница, цифры, _ . -)','UPLOAD');if(!g){e.target.value='';return}const fd=new FormData();fd.append('file',f);
 try{const r=await api('/api/upload?group='+encodeURIComponent(g),{method:'POST',body:fd});await reloadGroups(g);toast(`Загружено ${r.dialogs} диалогов — запускаю оценку`);runJob({group:g})}catch(er){toast('Ошибка: '+er.message)}e.target.value=''};
$('#mPaste').onclick=()=>{$('#pop').classList.add('hide');$('#pasteM').classList.remove('hide');$('#paste').focus()};
$('#btnPaste').onclick=async()=>{try{const r=await api('/api/paste',J({text:$('#paste').value,channel:$('#pch').value,language:$('#plang').value,group:'PASTE'}));$('#pasteM').classList.add('hide');await reloadGroups('PASTE');runJob({group:'PASTE',ids:[r.conversation_id]})}catch(e){toast('Ошибка: '+e.message)}};
$('#mRescore').onclick=async()=>{$('#pop').classList.add('hide');try{await api('/api/rescore/'+encodeURIComponent(G),{method:'POST'});await loadGroup(G,true);toast('Итоги пересчитаны')}catch(e){toast('Ошибка: '+e.message)}};
// ---------- checklist editor ----------
async function loadAllCl(){for(const c of await api('/api/checklists'))CLS[c.id]=(await api('/api/checklists/'+c.id)).checklist}
async function loadCl(id){await loadAllCl();const ls=Object.keys(CLS);$('#clSel').innerHTML=ls.map(x=>`<option>${esc(x)}</option>`).join('');const w=id||CL?.id;if(w&&CLS[w])$('#clSel').value=w;else $('#clSel').value=ls[0];CL=JSON.parse(JSON.stringify(CLS[$('#clSel').value]));renderCl()}
$('#clSel').onchange=()=>{CL=JSON.parse(JSON.stringify(CLS[$('#clSel').value]));renderCl()};
function renderCl(){$('#clId').value=CL.id;$('#clName').value=CL.name||'';$('#clCh').value=CL.channel||'chat';$('#clDesc').value=CL.description||'';$('#clAss').value=JSON.stringify(CL.assumptions||{});
 const ws=CL.criteria.reduce((a,c)=>a+(c.gross?0:+c.weight||0),0);$('#wsum').textContent=`${CL.criteria.length} критериев · сумма весов ${ws} (итог всегда нормируется к 100; грубые критерии в балл не входят и помечаются отдельно)`;
 $('#clCrit').innerHTML=CL.criteria.map((c,i)=>`<div class="ccard"><div class="r1">
  <label class="f">№<input value="${esc(c.id)}" onchange="CL.criteria[${i}].id=this.value"></label>
  <label class="f">Название<input value="${esc(c.name)}" onchange="CL.criteria[${i}].name=this.value"></label>
  <label class="f" title="Код критерия в results.json (v1.0); пусто — официальный код или №">Код<input value="${esc(c.code||'')}" placeholder="${esc(({'01':'contact','02':'discovery','03':'identification','04':'consultation','05':'empathy','06':'dialogue','07':'closure','08':'critical'})[c.id]||c.id)}" onchange="if(this.value.trim())CL.criteria[${i}].code=this.value.trim();else delete CL.criteria[${i}].code"></label>
  <label class="f">Тип<select onchange="CL.criteria[${i}].type=this.value;renderCl()"><option value="binary" ${c.type=='binary'?'selected':''}>Да / Нет</option><option value="scale" ${c.type=='scale'?'selected':''}>Шкала</option></select></label>
  <label class="f">Шкала${c.type=='scale'?`<input value="${(c.scale||[]).join(',')}" onchange="CL.criteria[${i}].scale=this.value.split(',').map(Number)">`:'<input disabled value="—">'}</label>
  <label class="f">Вес<input type="number" value="${c.weight}" onchange="CL.criteria[${i}].weight=+this.value;renderCl()"></label>
  <label class="ck"><input type="checkbox" ${c.allow_na?'checked':''} onchange="CL.criteria[${i}].allow_na=this.checked">Допускает н/п</label>
  <label class="ck"><input type="checkbox" ${c.gross?'checked':''} onchange="CL.criteria[${i}].gross=this.checked">Грубая ошибка</label>
  <div style="display:flex;gap:4px;padding-bottom:1px;justify-content:flex-end">${i>0?`<button class="btn" onclick="mv(${i},-1)" title="Выше">↑</button>`:''}<button class="btn" onclick="delCrit(${i})" title="Удалить">✕</button></div></div>
  <div class="r2"><label class="f">Рубрика<textarea rows="6" onchange="CL.criteria[${i}].rubric=this.value">${esc(c.rubric)}</textarea></label>
  <label class="f">Правила решения<textarea rows="6" onchange="CL.criteria[${i}].rules=this.value">${esc(c.rules)}</textarea></label></div></div>`).join('')}
function mv(i,d){const c=CL.criteria.splice(i,1)[0];CL.criteria.splice(i+d,0,c);renderCl()}
$('#btnAddCrit').onclick=()=>{CL.criteria.push({id:String(CL.criteria.length+1).padStart(2,'0'),name:'Новый критерий',type:'binary',weight:10,allow_na:false,gross:false,rubric:'YES — ...\nNO — ...',rules:''});renderCl()};
$('#btnClNew').onclick=()=>{CL=JSON.parse(JSON.stringify(CL));CL.id=CL.id+'_copy';renderCl();toast('Создана копия — измените идентификатор и сохраните')};
$('#btnYaml').onclick=async()=>{const b=$('#yamlBox');b.classList.toggle('hide');const on=!b.classList.contains('hide');$('#clForm').classList.toggle('hide',on);$('#btnYaml').textContent=on?'Форма':'Редактор YAML';if(on){const r=await api('/api/checklists/'+encodeURIComponent($('#clSel').value));$('#yaml').value=r.yaml}};
$('#btnClSave').onclick=async()=>{try{let body;if(!$('#yamlBox').classList.contains('hide'))body={yaml:$('#yaml').value};else{CL.id=$('#clId').value;CL.name=$('#clName').value;CL.channel=$('#clCh').value;CL.description=$('#clDesc').value;CL.assumptions=JSON.parse($('#clAss').value||'{}');body={checklist:CL}}
 const r=await api('/api/checklists/'+encodeURIComponent(CL.id||$('#clSel').value),{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});$('#clMsg').textContent=`Сохранено: ${r.id}. Теперь «Ещё → Переоценить» (только изменённые критерии) или «Пересчитать веса».`;toast('Чек-лист сохранён');await loadCl(r.id)}catch(e){$('#clMsg').textContent='Ошибка: '+e.message}};
init().catch(e=>{$('#ovw').innerHTML=`<div class="card empty">Ошибка загрузки: ${esc(e.message)}</div>`});
async function delCrit(i){if(await ask('Удалить критерий?',null)){CL.criteria.splice(i,1);renderCl()}}

// ---------- appeals (two-step: open -> resolved; API /api/appeals) ----------
const apFor=(conv,cid)=>APS.find(a=>a.status=='open'&&a.conversation_id==conv&&a.criterion_id==cid);
async function refreshAps(){try{const r=await api('/api/appeals?group='+encodeURIComponent(G));APS=r.items;const n=r.open;$('#apCnt').textContent=n;$('#apCnt').classList.toggle('hide',!n)}catch(e){APS=[]}}
function apBlock(cid,c){const a=apFor(CUR,cid);const def=cdef(DOC?.dialogs.find(x=>x.conversation_id==CUR)?.checklist_id,cid);
 const done=APS.filter(x=>x.status=='resolved'&&x.conversation_id==CUR&&x.criterion_id==cid).map(x=>`<div class="apn apdone">⚑ Апелляция${x.author?' ('+esc(x.author)+')':''}: «${esc(x.comment)}» · исходная оценка ${badge(x.value_at_appeal,def)} · решение: <b>${x.resolution.decision=='changed'?'изменено':'подтверждено'}</b> — ${esc(x.resolution.reason)} · итоговая ${badge(x.resolution.value,def,x.resolution.decision=='changed'?'hum':'')}</div>`).join('');
 if(a)return done+ `<div class="apn">⚑ Апелляция открыта${a.author?' ('+esc(a.author)+')':''}: «${esc(a.comment)}» <span class="lnk" onclick="view('appeals')">к очереди</span></div>`;
 if(APOPEN!=cid)return done;return done+`<div class="ovf apf"><textarea id="apc_${cid}" rows="2" placeholder="Почему вы не согласны с оценкой «${esc(VL[c.value]??c.value)}»? (обязательно)"></textarea>
 <input id="apa_${cid}" placeholder="Имя" style="width:110px;flex:0 0 110px;min-width:0" value="${esc(localStorage.reviewer||'')}">
 <button class="btn pri" id="apSend_${cid}" onclick="sendAp('${cid}')">Отправить апелляцию</button><button class="btn" onclick="APOPEN=null;renderDialog()">Отмена</button></div>`}
async function sendAp(cid){const cm=$('#apc_'+cid).value.trim();if(!cm){toast('Комментарий обязателен');$('#apc_'+cid).focus();return}
 localStorage.reviewer=$('#apa_'+cid).value;const st=$('#crits').scrollTop;
 try{await api('/api/appeals',J({group:G,run:RUN,conversation_id:CUR,criterion_id:cid,comment:cm,author:$('#apa_'+cid).value}));APOPEN=null;await refreshAps();renderDialog();$('#crits').scrollTop=st;toast('Апелляция открыта')}catch(e){toast('Ошибка: '+e.message)}}
$$('#apSeg a').forEach(a=>a.onclick=()=>{APSTAT=a.dataset.s;$$('#apSeg a').forEach(x=>x.classList.toggle('on',x==a));loadAppeals()});
async function loadAppeals(){let r;try{r=await api(`/api/appeals?group=${encodeURIComponent(G)}&fragment=true${APSTAT?'&status='+APSTAT:''}`)}catch(e){$('#apList').innerHTML=`<div class="card empty">${esc(e.message)}</div>`;return}
 $('#apMeta').textContent=`группа ${G} · открыто ${r.open}`;$('#apCnt').textContent=r.open;$('#apCnt').classList.toggle('hide',!r.open);
 if(!r.items.length){$('#apList').innerHTML='<div class="card empty">Апелляций нет.</div>';return}
 $('#apList').innerHTML=r.items.slice().reverse().map(apCard).join('')}
function apCard(a){const def=cdef(a.checklist_id,a.criterion_id);const opts=(def?.type=='scale'?(def.scale||[]).map(String):['YES','NO']).concat(def?.allow_na?['NA']:[]);
 const res=a.resolution;
 return `<div class="card apcard" id="ap_${a.id}"><div class="aph"><b>${esc(a.criterion_id)}</b> ${esc(a.criterion_name)} · <span class="lnk mono" onclick="openDialog('${esc(a.conversation_id)}')">${esc(a.conversation_id)}</span>
  <span class="sp"></span><span class="tag ${a.status=='open'?'bad':'soft'}">${a.status=='open'?'открыта':res.decision=='changed'?'решена · изменено':'решена · подтверждено'}</span></div>
  <div class="apg"><div><div class="mut sm">Вердикт AI</div>${badge(a.ai_value,def)}${String(a.value_at_appeal)!=String(a.ai_value)?` → ${badge(a.value_at_appeal,def,'hum')}`:''} <span class="mut sm">итог на момент апелляции: ${a.total_at_appeal!=null?a.total_at_appeal.toFixed(2):'—'}</span>
   <div class="js">${esc(a.ai_justification)}</div>${(a.ai_evidence||[]).map(e=>`<div class="evq"><b>#${e.turn}</b>«${esc(e.quote)}»</div>`).join('')}${a.missing_where?`<div class="meta">Ожидалось: ${esc(a.missing_where)}</div>`:''}
   <div class="apcm"><div class="mut sm">Комментарий${a.author?' · '+esc(a.author):''} · ${esc(a.created_at.replace('T',' '))}</div>«${esc(a.comment)}»</div></div>
  <div><div class="mut sm">Фрагмент диалога (нумерация с №1)</div><div class="apfr">${(a.fragment||[]).map(t=>`<div class="turn ${esc(t.speaker)} ${t.evidence?'ev':''}"><div class="n">${t.turn}</div><div class="bub"><div class="who">${SPK[t.speaker]||esc(t.speaker)}</div><div class="tx">${esc(t.text)}</div></div></div>`).join('')}</div></div></div>
  ${a.status=='open'?`<div class="ovf apr"><label><input type="radio" name="apd_${a.id}" value="confirm" checked onchange="$('#apv_${a.id}').disabled=true"> Подтвердить</label>
   <label><input type="radio" name="apd_${a.id}" value="change" onchange="$('#apv_${a.id}').disabled=false"> Изменить на</label>
   <select id="apv_${a.id}" disabled>${opts.map(x=>`<option value="${x}" ${String(a.value_at_appeal)==x?'selected':''}>${VL[x]??x}</option>`).join('')}</select>
   <input id="apr_${a.id}" placeholder="Причина решения (обязательно)"><input id="aprv_${a.id}" placeholder="Проверяющий" style="width:130px;flex:0 0 130px;min-width:0" value="${esc(localStorage.reviewer||'')}">
   <button class="btn pri" id="apOk_${a.id}" onclick="resolveAp('${a.id}')">Решить</button></div>`:
  `<div class="hn">Решение: <b>${res.decision=='changed'?'изменено':'подтверждено'}</b> ${badge(res.previous_value,def)} → ${badge(res.value,def,res.decision=='changed'?'hum':'')} — ${esc(res.reason)}${res.reviewer?' ('+esc(res.reviewer)+')':''}</div>`}
  <details><summary>Журнал (${a.audit.length})</summary>${a.audit.map(x=>`<div class="mut sm">${esc(x.at.replace('T',' '))} · ${esc(x.who||'—')} · ${esc(x.action)}${x.comment?' · «'+esc(x.comment)+'»':''}${x.reason?' · '+esc(x.reason):''}${x.to!==undefined?` · ${esc(x.from)} → ${esc(x.to)}`:''}</div>`).join('')}</details></div>`}
async function resolveAp(id){const d=document.querySelector(`input[name=apd_${id}]:checked`).value,rs=$('#apr_'+id).value.trim();
 if(!rs){toast('Причина обязательна');$('#apr_'+id).focus();return}localStorage.reviewer=$('#aprv_'+id).value;
 try{const r=await api(`/api/appeals/${id}/resolve`,J({group:G,decision:d,value:d=='change'?$('#apv_'+id).value:null,reason:rs,reviewer:$('#aprv_'+id).value}));
  const i=DOC?DOC.dialogs.findIndex(x=>x.conversation_id==r.dialog.conversation_id):-1;if(i>=0&&(!r.appeal.run||r.appeal.run==RUN))DOC.dialogs[i]=r.dialog;
  if(CUR==r.dialog.conversation_id)renderDialog();toast(`Апелляция решена · итог диалога: ${r.total!=null?r.total.toFixed(2):'—'}`);await refreshAps();renderTable();renderOverview();await loadAppeals()}catch(e){toast('Ошибка: '+e.message)}}
