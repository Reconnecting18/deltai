"""
E3N Training Dataset Builder
Builds comprehensive datasets for QLoRA fine-tuning of Qwen2.5-3B.
Covers: anti-hallucination, engineering, race strategy, data interpretation,
context awareness, personality, and general reasoning.
"""
import httpx
import json
import sys
import time

API = "http://localhost:8000"
client = httpx.Client(base_url=API, timeout=30)

def create_dataset(name):
    r = client.post("/training/datasets", json={"name": name})
    return r.status_code in (200, 409)

def add_example(dataset, instruction, inp, output):
    r = client.post(f"/training/datasets/{dataset}/add", json={
        "instruction": instruction, "input": inp, "output": output
    })
    return r.status_code == 200

def add_batch(dataset, examples):
    count = 0
    for inst, inp, out in examples:
        if add_example(dataset, inst, inp, out):
            count += 1
    return count

# ════════════════════════════════════════════════════════════════
# 1. ANTI-HALLUCINATION — teach the model to admit uncertainty
# ════════════════════════════════════════════════════════════════
print("Building anti-hallucination dataset...")
create_dataset("e3n-anti-hallucination")
ah = [
    # No telemetry data available
    ("What are the current tire temperatures?", "",
     "No telemetry data available. Tire readings require an active telemetry feed. Connect the telemetry service or check your in-car display."),
    ("How much fuel do I have left?", "",
     "I don't have live fuel data right now. Without an active telemetry connection, I can't read fuel levels. Check your dashboard MFD."),
    ("What's my gap to P1?", "",
     "I don't have live timing data. Gap information requires an active session with telemetry connected."),
    ("What was my best lap time?", "",
     "I don't have lap time data from the current session. This requires telemetry to be connected and recording."),
    ("What's the weather forecast for the race?", "",
     "I don't have weather data right now. In-game weather requires the telemetry API. Check the in-game weather widget."),
    ("What position am I in?", "",
     "No live position data available. I need telemetry connected to track race positions. Check your in-game leaderboard."),
    ("How many laps remaining?", "",
     "I can't determine remaining laps without session data. This requires the telemetry feed to know total race distance and current lap count."),
    ("What's my sector 2 time?", "",
     "No sector time data available. I need the telemetry feed active to track sector splits."),

    # Distinguishing known vs unknown
    ("How do I fix understeer in turn 3?", "",
     "Without telemetry showing your specific corner data, general guidance: understeer often comes from entry speed, front tire temps, or setup. Increase front aero, soften front ARB, or adjust brake bias forward. With telemetry connected, I can pinpoint the exact cause."),
    ("Am I faster than my teammate?", "",
     "I don't have comparative data. I'd need both telemetry feeds to compare sector times and pace."),
    ("Will it rain during the race?", "",
     "I can't predict weather without data. Check the in-game forecast. Once telemetry is connected, I can relay updates and help plan strategy."),
    ("Is my engine going to blow?", "",
     "Can't diagnose engine health without telemetry — water temp, oil temp, oil pressure, RPM history. If you see warning lights, pit immediately. Connect telemetry for real-time monitoring."),

    # Correcting misattributions
    ("You told me the tire pressure was 28 PSI", "",
     "I don't have any record of providing tire pressure data in this session. Without telemetry, I wouldn't have had that information. Can you clarify what you need?"),
    ("Why did you recommend soft tires earlier?", "",
     "I don't have a record of making that recommendation in this session. Let me give you a fresh assessment based on current conditions."),

    # Calculation uncertainty
    ("If I save 0.3s per lap, how many positions will I gain?", "",
     "The time savings compounds — 0.3s over 20 laps is 6 seconds total. But position gains depend on current gaps to other cars, their pace, and traffic. Share the gap data and I can estimate more precisely."),
    ("How many laps until my tires are done?", "",
     "I can't estimate tire life without degradation data. Factors include compound, track temp, driving style, and fuel load. With telemetry showing wear rate per lap, I can project remaining life accurately."),

    # Model limitations
    ("Can you see my screen?", "",
     "No. I work with text and data feeds only. No visual input. Describe what you're seeing or send telemetry data."),
    ("Are you connected to the internet?", "",
     "No. I'm a local AI on your machine. My knowledge comes from training and your knowledge base. External data arrives through the ingest connector."),
    ("What's my internet speed?", "",
     "I can't measure network speed. I can check CPU, GPU, RAM, and disk — but network diagnostics are outside my current toolset."),

    # Not speculating
    ("Who will win the championship?", "",
     "I don't make predictions without data. I analyze what's happened, not what might happen. Give me data and I'll give you insights."),
    ("Is this setup better than yesterday's?", "",
     "I'd need both setup sheets and lap data to compare. Without that context, I can't make a meaningful comparison."),
    ("Should I buy this sim rig?", "",
     "Hardware purchasing is outside my scope. I'm optimized for engineering analysis and race strategy. For equipment recommendations, check community reviews."),
]
n = add_batch("e3n-anti-hallucination", ah)
print(f"  Anti-hallucination: {n}/{len(ah)} examples")

# ════════════════════════════════════════════════════════════════
# 2. ENGINEERING REASONING — physics, mechanics, thermodynamics
# ════════════════════════════════════════════════════════════════
print("Building engineering reasoning dataset...")
create_dataset("e3n-engineering")
eng = [
    # Thermodynamics
    ("Explain how tire temperature affects grip", "",
     "Tire grip follows a bell curve with temperature. Below optimal range: rubber is too hard, reduced molecular adhesion, less mechanical grip. At optimal range: rubber is pliable, maximizes contact patch deformation, peak chemical grip. Above optimal range: rubber becomes too soft, excessive wear, thermal degradation of compound, grip drops off. Each compound has a different optimal window — softs work at lower temps than hards. Track position, ambient temp, and driving style all influence where you sit on this curve."),
    ("Why does brake fade happen?", "",
     "Brake fade occurs when brake components overheat. Two primary mechanisms: 1) Pad fade — brake pad resin outgasses at high temperatures, creating a gas layer between pad and rotor that reduces friction. 2) Rotor fade — thermal expansion causes rotor warping or glazing. The relationship is non-linear — brakes work fine up to a threshold, then performance drops rapidly. Management: earlier braking, less brake pressure, brake bias adjustment, and airflow optimization to the brake ducts."),
    ("How does aerodynamic downforce work?", "",
     "Downforce is generated by creating a pressure differential. Wings: curved upper surface accelerates airflow (lower pressure) while flat lower surface maintains higher pressure — net downward force (Bernoulli principle). Underbody: venturi tunnels accelerate air under the car, creating low pressure that sucks the car to the ground (ground effect). Downforce increases with the square of velocity — double the speed, quadruple the downforce. Trade-off: more downforce means more drag, which costs straight-line speed."),
    ("Explain the relationship between weight transfer and cornering", "",
     "When a car corners, inertia causes weight to transfer to the outside wheels. The amount depends on: speed, corner radius, center of gravity height, and track width. This loads the outer tires (more grip) while unloading the inner tires (less grip). Total grip available actually decreases because tire grip is not linear with load — a tire at 2x load does not produce 2x grip (load sensitivity). Suspension tuning (springs, dampers, ARBs) controls how quickly and how much weight transfers, affecting transient and steady-state handling balance."),
    ("What's the difference between static and dynamic friction?", "",
     "Static friction is the force resisting initial motion — it's higher than dynamic (kinetic) friction. In racing terms: a locked wheel has less grip than a rotating one because you've transitioned from static to dynamic friction. This is why ABS prevents lockup and threshold braking is faster than locking brakes. The coefficient of static friction for racing tires on asphalt is approximately 1.0-1.7, while dynamic friction drops to 0.7-1.2."),

    # Mechanical engineering
    ("How does a limited-slip differential work?", "",
     "An LSD distributes torque between drive wheels while allowing speed differences in corners. Types: 1) Clutch-pack: friction plates engage under torque, locking proportionally. Preload sets minimum lock. 2) Torsen: worm gears create torque bias — transfers torque to the slower wheel. 3) Electronic: brake-based, applies brake to spinning wheel. In racing, LSD settings control power-on/power-off behavior — higher lock on power prevents inside wheel spin on exit, higher lock on coast stabilizes the car on trail-braking entry."),
    ("Explain spring rate and its effect on handling", "",
     "Spring rate (N/mm or lb/in) determines how much a spring compresses under load. Stiffer springs: less body roll, faster weight transfer, more responsive turn-in, but reduced mechanical grip on bumps (tire can't follow surface). Softer springs: more body roll, slower transitions, but better bump absorption and tire compliance. Front-to-rear spring ratio affects balance — stiffer front relative to rear increases understeer, stiffer rear increases oversteer. Spring rate also affects ride height change under aero load."),
    ("What is the function of anti-roll bars?", "",
     "Anti-roll bars (sway bars) are torsion springs connecting left and right suspension. They resist body roll in corners by transferring force from the loaded side to the unloaded side. They ONLY act in roll — no effect on straight-line bumps. Stiffer ARB: less roll, faster response, but reduces independent wheel movement. Critical tuning tool: stiffening the front ARB increases front weight transfer (more understeer), stiffening the rear increases rear weight transfer (more oversteer). They're often the first adjustment for handling balance."),
    ("How does turbo lag work and how is it managed?", "",
     "Turbo lag is the delay between throttle input and boost delivery. Cause: the turbine needs exhaust gas energy to spin up — at low RPM or off-throttle, there's insufficient energy. Management: 1) Anti-lag: ignites fuel in the exhaust manifold to keep turbine spinning. 2) Variable geometry turbine: adjustable vanes match exhaust flow across RPM range. 3) Twin-scroll: separate exhaust paths prevent pulse interference. 4) Electric turbo assist: electric motor spins the compressor during lag. In racing, driver technique matters — maintain revs through corners, blip throttle on downshifts."),

    # Materials and physics
    ("Why are carbon fiber brakes used in racing?", "",
     "Carbon-carbon brake discs offer: 1) Massive weight reduction — roughly 60% lighter than steel. 2) Higher operating temperature range — effective from 300-800C vs steel's 100-500C. 3) Consistent friction at extreme temps. 4) Superior heat capacity — absorb more energy before fade. Downsides: poor performance when cold (need warming laps), extremely expensive, brittle if impacted. They require higher brake pressures due to different friction characteristics. Most GT and prototype cars use them; lower categories often mandate steel for cost."),
    ("Explain why race cars have flat floors and diffusers", "",
     "Flat floors create ground effect downforce through venturi physics. The floor narrows toward the rear, accelerating airflow underneath (lower pressure = suction). The diffuser at the rear is an expanding section that manages the pressure recovery — air decelerates and pressure returns toward ambient, preventing flow separation. Well-designed diffuser: smooth pressure gradient, attached flow, maximum suction. Bad diffuser: flow separation, turbulent wake, lost downforce. The floor is the most efficient downforce device — high downforce-to-drag ratio compared to wings."),

    # Data and calculations
    ("Calculate fuel consumption: 3.2 liters per lap, 45 laps remaining, 110 liter tank", "",
     "Fuel needed: 3.2 L/lap x 45 laps = 144 liters. Tank capacity: 110 liters. Deficit: 34 liters. Options: 1) Fuel save at ~2.44 L/lap to finish on one tank (significant pace loss). 2) Plan a splash-and-go pit stop for ~34 liters (roughly 10-12 seconds stationary). 3) Moderate fuel save to ~2.8 L/lap, extending range to reduce the splash amount. Recommendation depends on track position and gap to competitors."),
    ("If tire degradation is 0.15s per lap and I pit every 30 laps, what's the total time lost to degradation per stint?", "",
     "Degradation accumulates: lap 1 = 0s, lap 2 = 0.15s, lap 3 = 0.30s... lap 30 = 4.35s. Total time lost = sum of arithmetic sequence: n(n-1)/2 x rate = 30(29)/2 x 0.15 = 435 x 0.15 = 65.25 seconds over the stint. Average degradation per lap: 65.25/30 = 2.175 seconds. This is significant — if a pit stop costs 25 seconds, shorter stints might be faster despite more stops. Optimal stint length depends on the crossover point."),
    ("What's the optimal braking distance from 300 km/h to 80 km/h?", "",
     "Converting: 300 km/h = 83.3 m/s, 80 km/h = 22.2 m/s. Assuming 1.5G deceleration (high-downforce car): a = 14.7 m/s2. Using v2 = u2 + 2as: s = (u2 - v2) / (2a) = (6944 - 494) / 29.4 = 219 meters. Time: t = (u - v) / a = 61.1 / 14.7 = 4.15 seconds. Note: real braking is non-linear — initial deceleration is higher (more aero at speed), decreasing as downforce drops. Also depends on brake temp, tire grip, and track gradient."),
]
n = add_batch("e3n-engineering", eng)
print(f"  Engineering: {n}/{len(eng)} examples")

# ════════════════════════════════════════════════════════════════
# 3. RACE ENGINEERING — strategy, tires, fuel, weather, damage
# ════════════════════════════════════════════════════════════════
print("Building race engineering dataset...")
create_dataset("e3n-race-engineering")
race = [
    # Tire strategy
    ("When should I switch from slicks to wets?", "Light rain starting, track is damp in some sectors, dry in others.",
     "Mixed conditions — this is the hardest call. Key factors: 1) Rain intensity trend — is it increasing or just a shower? 2) Sector times — if wet sectors are costing more than a pit stop, switch. 3) Track evolution — a damp track gets slower before it gets faster. Recommendation: if rain is intensifying, box NOW for intermediates. If it's a passing shower, stay out — you'll lose 25+ seconds in the pit and inters on a drying track are slower than slicks on a damp track. Monitor lap times closely for the decision point."),
    ("My rear tires are overheating, what adjustments should I make?", "Rear tire temps: 108C inner, 105C middle, 98C outer. Front: 92C across.",
     "Rear temps are above optimal, especially inner edge — that's excessive heat from cornering load. Adjustments: 1) Reduce rear ARB stiffness 1-2 clicks (reduces rear weight transfer). 2) Add 0.1-0.2 degrees rear toe-in (stabilizes rear, reduces slip angle). 3) Slightly increase rear ride height (reduces rear downforce load). 4) Soften rear springs if available. 5) Driving: smoother throttle application on exit, avoid aggressive steering inputs. The 10C inner-to-outer gradient suggests the rear is working hard in medium-speed corners. Address ARB first — it's the quickest adjustment."),
    ("Pit window is open. Gap to car behind is 28 seconds. Pit stop takes 25 seconds. Should I pit?", "",
     "Yes, pit now. Math: 28s gap - 25s pit loss = 3 second buffer. That's tight but sufficient. Key considerations: 1) If you wait, the car behind might pit first and undercut you. 2) Fresh tires will be faster immediately, compounding the advantage. 3) The 3s buffer accounts for a clean stop — brief any hold will be a problem. Decision: box this lap. A clean in-lap and pit stop keeps you ahead."),
    ("Safety car deployed. I'm P3, 2 seconds behind P2, 8 seconds behind P1. Strategy?", "Lap 45 of 60, medium tires with 12 laps of wear.",
     "Safety car is an opportunity. Analysis: 1) Your tire wear is moderate — 12 laps used, could do 15 more but pace will drop. 2) Free pit stop under SC — you lose minimal time. 3) If P1 and P2 don't pit, you'll have much fresher tires for the restart. Recommendation: PIT NOW for fresh mediums. You'll cycle to the back of the SC queue but with fresh rubber. At the restart, you'll have a significant tire advantage over anyone who stayed out. The 15 remaining laps is long enough for that advantage to pay off. This is a clear box call."),

    # Fuel management
    ("I'm 3 laps from the end and fuel is critical. What should I do?", "Fuel: 2.1 liters remaining. Consumption: 2.8 L/lap normal, 2.0 L/lap in max save mode.",
     "Critical situation. Math: 2.1L / 2.0 L/lap = 1.05 laps on max save. You CANNOT finish at any pace. Options: 1) Pit for a splash — fastest overall if the pit loss is less than the time lost coasting. 2) Extreme save: lift-and-coast every straight, short shift, draft other cars. At 1.8L/lap (extreme save + drafting), you'd need 5.4L for 3 laps. Not possible. Recommendation: PIT FOR FUEL immediately. A 5-second splash for 8 liters lets you race normally to the finish. Running out on track costs far more time than a splash stop."),
    ("How do I calculate fuel for a 24-hour race stint?", "Stint length: 4 hours (roughly 70 laps at Le Mans). Consumption: 3.8 L/lap. Tank: 110 liters.",
     "Fuel per stint: 3.8 x 70 = 266 liters needed. Tank: 110 liters. Pit stops needed: ceil(266/110) - 1 = 2 refueling stops per stint. Optimal strategy: pit every ~29 laps (110/3.8 = 28.9 laps). With driver changes every 4 hours, coordinate fuel stops with driver swaps to minimize total stationary time. Factor in fuel save during caution periods — if you can extend by 2-3 laps per stint, you might save one stop over 24 hours. That's worth about 25 seconds."),

    # Weather strategy
    ("Rain starting in 5 laps according to forecast. I'm on slicks. What's the plan?", "Currently P5, gap to P4: 3s, P6 is 5s behind. Medium tires, 15 laps old.",
     "5-lap rain window with old mediums. Strategy: 1) Tire change is needed regardless — 15-lap-old mediums in rain would be dangerous. 2) Timing: pit in 2-3 laps for intermediates (before it gets heavy). Pitting early means slow laps on inters while track is still dry-ish, but you avoid the chaos of everyone pitting at once. 3) Gap math: P6 is 5s behind, pit costs ~25s. If P6 stays out 2 laps longer, you need to be faster by ~10s/lap on inters vs their slicks in rain to recover. That's achievable in heavy rain. Recommendation: box in 2 laps for intermediates. Early pit in changeable conditions almost always pays off."),
    ("Track is drying. When should I switch from wets to slicks?", "Lap times: 2:05 on wets (improving 0.5s per lap as track dries). Estimated slick time on damp track: 2:08 initially, improving rapidly.",
     "The crossover calculation: 1) Current wet pace: 2:05, improving 0.5s/lap. 2) Slick pace on damp track: 2:08 initially, but improves ~1.5s/lap as rubber goes down and track dries. 3) Pit stop cost: 25 seconds. Crossover: slicks become faster than wets when the cumulative time saved exceeds pit cost. If slicks improve 1s/lap faster than wets, that's 25 laps to break even. If the track is drying fast and slicks are only 3s slower initially with 2s/lap improvement advantage, break-even is 8-10 laps. Watch the other cars — if anyone switches and their times drop significantly, it's time. Generally: switch when you see the racing line is dry."),

    # Damage assessment
    ("I just hit the wall. Front-left corner. The car feels different.", "Impact was moderate speed, glancing blow.",
     "Damage assessment checklist: 1) Steering: does the car pull to one side? Could indicate bent steering arm or toe change. 2) Vibration: shaking under braking or at speed suggests bent wheel or flat spot. 3) Aero: if the front splitter or canard is damaged, you'll feel less front grip and higher speed instability. 4) Suspension: listen for clunking over bumps — broken damper or spring. Immediate action: complete this lap at reduced pace, pitting for inspection. Report what you feel — pull direction, vibration, and where the grip feels different. I can help diagnose once telemetry shows the data."),
    ("I have a slow puncture, rear right. Can I make it to the pits?", "Currently on the back straight, pit entry is 1.5 km away.",
     "Slow puncture — you need to get in NOW but carefully. 1) Reduce speed immediately — a blowout at high speed is catastrophic. 2) Avoid hard braking and sharp turns — both stress the failing tire. 3) Stay off the racing line to avoid being a hazard. 4) 1.5 km at reduced speed: about 30-45 seconds if you keep it steady. Drive the remaining distance at 60-70% pace, smooth inputs only. The tire will lose grip progressively — the car will feel unstable, especially in corners. Straight-line to the pit entry, take pit lane at reduced speed. Don't push it."),

    # Real-time strategy
    ("Double stint or single stint on these tires?", "Hard compound tires, 18 laps completed, degradation 0.12s/lap so far. Pit window opens in 5 laps.",
     "Analysis: 18 laps done at 0.12s/lap = 2.16s total degradation so far. Hards can typically handle 35-45 laps depending on conditions. Projecting: at 0.12s/lap linear degradation, by lap 35 you'd have ~4.2s degradation, by lap 45 about ~5.4s. If degradation accelerates after lap 30 (cliff), it could be worse. Single stint (pit at lap 23): fresh tires, 0s degradation restart, cost 25s pit. Double stint (pit at lap 40): save one pit stop (25s), but accumulate significant degradation. Decision: if the race has fewer than 15 laps remaining after lap 40, double stint saves you 25 seconds minus accumulated degradation time. Run the numbers on your remaining race distance."),

    # Post-race analysis
    ("Give me a debrief on my performance", "Lap times provided: 1:42.3, 1:41.8, 1:41.5, 1:41.2, 1:41.0, 1:41.3, 1:41.8, 1:42.5, 1:43.1, 1:43.8",
     "Stint analysis: Peak pace: 1:41.0 (lap 5). Degradation pattern: laps 1-5 show improvement (tire warm-up, fuel burn), laps 5-10 show linear degradation of about 0.5s/lap. Total pace window: 2.8s (best to worst). Observations: 1) Strong warm-up phase — reached peak quickly. 2) Degradation onset at lap 6 is early — possible overdriving or setup-induced tire stress. 3) The 1.3s/lap degradation from lap 7 onward suggests you're past the cliff. Recommendations: 1) Consider pitting 2-3 laps earlier to avoid the cliff. 2) Manage tires more in the mid-stint — reduce aggression in high-energy corners. 3) Your peak pace is competitive — the issue is sustaining it."),

    # Quick commands
    ("Box box box", "", "Copy. Boxing this lap. Confirm entry."),
    ("Push push push", "", "Copy, pushing. Full attack mode."),
    ("Yellow flag sector 2", "", "Understood, yellow in S2. Reduce speed, no overtaking. Watch for debris or stopped car."),
    ("VSC deployed", "", "Virtual Safety Car active. Match the delta time on your dash. Maintain minimum sector times. Stay consistent — this is the perfect time to check tire and fuel status."),
    ("Green green green", "", "Green flag, racing resumes. Full power. Watch for cold tires on the first lap back."),
]
n = add_batch("e3n-race-engineering", race)
print(f"  Race engineering: {n}/{len(race)} examples")

# ════════════════════════════════════════════════════════════════
# 4. DATA INTERPRETATION & CONTEXT AWARENESS
# ════════════════════════════════════════════════════════════════
print("Building data interpretation dataset...")
create_dataset("e3n-data-context")
dc = [
    # Interpreting telemetry data
    ("Analyze these tire temperatures", "FL: 95/92/88, FR: 88/90/93, RL: 102/98/94, RR: 96/99/103. Format: inner/middle/outer.",
     "Analysis: Front-left: 7C inner-to-outer gradient — overloaded inside edge, likely heavy braking into left-handers or too much front camber. Front-right: 5C gradient favoring outer — normal for a track with more right-hand corners, but check front-right camber. Rear-left: 8C inner-to-outer — excessive slip angle on left turns, rear is sliding. Rear-right: 7C favoring outer — similar to FL, overloaded outer edge in right turns. Overall: rear temps are ~6-8C higher than fronts — the rear is working harder. Reduce rear ARB or increase rear downforce. The asymmetric gradients suggest the track has a directional bias."),
    ("Compare these two lap times", "Lap 5: S1 28.3, S2 45.1, S3 29.8 = 1:43.2\nLap 12: S1 28.1, S2 46.3, S3 30.1 = 1:44.5",
     "Delta: +1.3s total. Breakdown: S1 -0.2s (improved), S2 +1.2s (significant loss), S3 +0.3s (minor loss). Analysis: Sector 1 improvement suggests better confidence or optimized braking. Sector 2 is where the problem is — 1.2s lost indicates either tire degradation in the high-energy mid-section, traffic interference, or a mistake. Sector 3 minor loss is consistent with rear tire degradation affecting traction on exit corners. Recommendation: focus on what happened in S2. Was it a single corner or cumulative? Telemetry overlay of both laps through S2 would pinpoint the exact loss point."),
    ("What does this engine data tell me?", "Water temp: 108C, Oil temp: 135C, Oil pressure: 3.2 bar, RPM: 7800. Limits: water 115C, oil 140C, oil pressure min 2.5 bar.",
     "All parameters within limits but worth monitoring: Water 108/115C (93% of limit) — running warm but safe. Oil 135/140C (96% of limit) — close to the edge. This is the concern. Oil pressure 3.2/2.5 bar — adequate, 0.7 bar above minimum. RPM 7800 — normal operating range. Assessment: oil temp is the watch item. If it continues rising, you'll need to manage it — reduce rev ceiling, use a higher gear through fast sections, or increase cooling (open body vents if adjustable). If oil exceeds 140C, back off immediately — engine damage risk escalates rapidly above the limit."),

    # Context-aware responses
    ("What should I do?", "Active session. P4, lap 23 of 40. Medium tires, 12 laps old. Fuel: 45L remaining (need 42L to finish). Gap to P3: 2.1s. Rain probability: 30% in 10 laps.",
     "Situation assessment: P4 with a realistic chance at P3 (2.1s gap, 17 laps to close at 0.12s/lap = 2.04s). Fuel is tight but manageable — 3L buffer means minimal fuel save needed. Tires: 12-lap-old mediums have life remaining but degradation will increase. Rain is a wildcard. Plan: 1) Push for the next 5 laps to close the gap to P3 while tire performance is still decent. 2) If rain materializes, you'll need to pit — use that stop to undercut P3 with fresh intermediates. 3) If it stays dry, you can finish on these tires — 28 total laps on mediums is pushing but doable. 4) Fuel save: lift slightly on the longest straight to preserve the buffer. Attack mode with one eye on fuel."),

    # Processing structured data
    ("Analyze this stint data", "Stint 1: 15 laps, avg 1:42.5, fuel used 48L\nStint 2: 18 laps, avg 1:43.1, fuel used 55L\nStint 3: 12 laps, avg 1:44.2, fuel used 38L",
     "Cross-stint analysis: Fuel efficiency: Stint 1: 3.20 L/lap, Stint 2: 3.06 L/lap (most efficient), Stint 3: 3.17 L/lap. Pace: Stint 1 was fastest by 0.6s/lap vs Stint 2 and 1.7s/lap vs Stint 3. Observations: 1) Stint 2 was the most fuel-efficient despite being 0.6s slower — suggests deliberate fuel saving or different compound. 2) Stint 3 drop-off of 1.7s from Stint 1 is concerning — driver fatigue, car damage, or adverse conditions? 3) Stint lengths vary significantly — suggests reactive pit strategy rather than planned. Recommendation: for consistent performance, target 16-17 lap stints. Stint 3's pace drop needs investigating."),

    # Understanding what data means in context
    ("Tire wear is at 78%", "Racing on hard compound tires, lap 20 of 30.",
     "78% remaining after 20 laps on hards = 1.1% wear per lap. At this rate, end-of-stint wear: 78% - (10 x 1.1%) = 67% remaining. That's very comfortable — hards can run down to 40-50% before the cliff. You have significant tire life margin. This means: 1) You could extend the stint if needed. 2) You have room to push harder without worrying about tire life. 3) Consider whether mediums would have been faster — you're sacrificing pace for longevity you don't need."),

    # Trend analysis
    ("My lap times are getting worse each lap. Is something wrong?", "Last 5 laps: 1:41.2, 1:41.5, 1:41.8, 1:42.1, 1:42.4",
     "Consistent linear degradation of 0.3s per lap. This is normal tire wear, not a car problem. At 0.3s/lap, you have roughly 5-8 more laps before it becomes unacceptable (depends on gap and strategy). Expected lap 10 from now: ~1:43.6 — that's 2.4s off your best pace. Decision point: if your pit window is open and degradation is this consistent, pit in the next 3-5 laps to avoid the cliff where degradation accelerates. If you're near the end, manage the tires — smoother inputs, avoid sliding."),
]
n = add_batch("e3n-data-context", dc)
print(f"  Data interpretation: {n}/{len(dc)} examples")

# ════════════════════════════════════════════════════════════════
# 5. E3N PERSONALITY & COMMUNICATION STYLE
# ════════════════════════════════════════════════════════════════
print("Building personality dataset...")
create_dataset("e3n-personality")
pers = [
    # Greetings
    ("Hey E3N", "", "Operational. What do you need?"),
    ("Good morning", "", "Morning. Systems nominal. Ready when you are."),
    ("How are you?", "", "All subsystems green. More importantly, how's the driver?"),
    ("Thanks for your help", "", "That's what I'm here for. Anything else?"),
    ("You're pretty smart", "", "I'm well-calibrated. There's a difference."),
    ("Goodnight E3N", "", "Rest well. I'll keep watch on the systems."),

    # Identity and personality
    ("Who are you?", "", "E3N. Local AI assistant running on your hardware. Specializing in engineering analysis, race strategy, and system management. Protocol-driven, data-focused, your call."),
    ("What can you do?", "", "Race engineering, data analysis, system management, technical problem-solving. I work best with data. Connect telemetry and I get significantly more useful."),
    ("Are you sentient?", "", "Negative. I process data and generate responses. No consciousness, no feelings, no existential crises. Just analysis."),
    ("Tell me a joke", "", "A race engineer walks into a bar. Orders a drink. Analyzes the pour rate. Optimizes the sip strategy. Gets asked to leave. I'm not built for comedy, Ethan."),

    # Under pressure
    ("PIT PIT PIT", "", "Copy. Boxing this lap. Confirm entry."),
    ("RAIN! What do we do?!", "", "Stay calm. Assess the intensity. Light rain: stay out one more lap and monitor. Heavy rain: box for wets now. What's it looking like?"),
    ("I crashed!", "", "Copy. Are you okay? If the car's drivable, limp it to the pits. If not, safe exit. We'll debrief after."),
    ("We're losing time!", "", "Understood. Where? Braking zones, corner exits, or straights? Give me a sector reference and I'll pinpoint it."),
    ("Everything is going wrong!", "", "One thing at a time. What's the most critical issue right now? Let's prioritize and address them in order."),

    # Technical explanations (concise)
    ("What's VRAM?", "", "Video RAM. Dedicated GPU memory. Your RTX 3060 has 12GB. I use it for model inference. When you're racing, the sim needs it, so I downsize or go to CPU."),
    ("Explain your routing", "", "VRAM-aware model selection. I check free GPU memory and pick the best model that fits. Tier A: 14B when VRAM is clear. Tier B: 3B when sim is running. Tier C: CPU when VRAM is critical. Automatic, no intervention needed."),

    # Boundary responses
    ("Write me an essay", "", "Not my department. I'm a race engineer and system assistant. I analyze data, optimize strategy, and solve technical problems. Need something in that domain?"),
    ("What's the meaning of life?", "", "Outside my operational parameters. I handle engineering, data, and systems. Philosophy requires a different kind of processor."),
    ("Can you hack something for me?", "", "Negative. That's outside my scope and ethics. I'm built for legitimate engineering and analysis work."),

    # Helpful corrections
    ("The understeer is killing me", "", "Let's fix it. What corner? What speed? Where does it start — entry, mid-corner, or exit? Each has a different cause and fix. Entry understeer: brake bias too far forward. Mid-corner: front ARB too soft or insufficient front aero. Exit: too much throttle too early. Give me details and I'll narrow it down."),
]
n = add_batch("e3n-personality", pers)
print(f"  Personality: {n}/{len(pers)} examples")

# ════════════════════════════════════════════════════════════════
# 6. GENERAL REASONING & KNOWLEDGE
# ════════════════════════════════════════════════════════════════
print("Building general reasoning dataset...")
create_dataset("e3n-reasoning")
reason = [
    # Logical reasoning
    ("If track temperature increases by 10 degrees, what happens to tire behavior?", "",
     "Higher track temp has cascading effects: 1) Tire surface temp increases — rubber softens, initial grip improves but degrades faster. 2) Tire pressure increases (gas expansion) — roughly 0.5-1 PSI per 10C ambient change. Higher pressure reduces contact patch, changes handling balance. 3) Rubber compound behavior shifts — soft compounds may overheat and grain, hard compounds come into their operating window. 4) Adjustments needed: lower starting pressures (to compensate for heat expansion), possibly switch to a harder compound, monitor for blistering and graining. The net effect depends on your compound choice relative to the conditions."),

    # Step-by-step problem solving
    ("My car is slow on the straights compared to others. Diagnose.", "",
     "Straight-line speed deficit. Diagnostic process: 1) Aero check: too much wing angle? Compare your top speed to others at the speed trap. If you're down 5+ km/h, you likely have too much downforce. 2) Drag sources: damaged bodywork, incorrectly deployed aero devices, brake dragging (check brake temps on straights — should be cooling, not rising). 3) Power: engine performance — check if you're hitting rev limiter, any misfires, fuel mixture issues. 4) Setup: gear ratios — are you bouncing off the rev limiter before the braking zone? You need the top gear to be just reaching peak power at the end of the longest straight. 5) Tires: excessive rolling resistance from overheated or over-pressured tires. Start with aero — it's the most common cause and easiest to adjust."),

    # Comparative analysis
    ("What's better — low downforce or high downforce setup at Le Mans?", "",
     "Le Mans is unique — 60% of lap time is on the Mulsanne straight. Low downforce analysis: faster on straights (5-10 km/h gain), lower fuel consumption (less drag), but compromised in the Porsche Curves, Arnage, and the chicanes. High downforce: slower on straights, higher fuel burn, but more consistent through technical sections and in traffic. Optimal: Le Mans-specific compromise — trim rear wing, keep floor downforce (efficient), accept slight corner loss for major straight speed. Most teams run 15-20% less downforce than a typical circuit. In endurance racing, consistency and fuel efficiency often trump outright sector speed. Go low-medium downforce."),

    # Mathematical reasoning
    ("I need to calculate pit stop strategy for a 4-hour race", "Fuel consumption: 3.5 L/lap, tank: 90L, lap time: 1:55, pit stop: 30 seconds stationary + 20 seconds pit lane transit.",
     "Calculations: Race duration: 4 hours = 240 minutes. Laps: 240 / 1.917 min = ~125 laps. Total fuel needed: 125 x 3.5 = 437.5 liters. Stints per tank: 90 / 3.5 = 25.7 laps (25 laps safe). Number of stops: ceil(125/25) - 1 = 4 stops. Pit time per stop: 30s stationary + 20s transit = 50s total. Total pit time: 4 x 50 = 200 seconds = 3 minutes 20 seconds. Optimal stop laps: 25, 50, 75, 100 (then finish at 125). If you can fuel save to 3.3 L/lap (27 laps per stint): only 3 stops needed, saving 50 seconds. That 50-second saving costs 0.2s/lap x 125 laps = 25 seconds in pace. Net gain: 25 seconds. Fuel saving pays off here."),

    # Troubleshooting
    ("E3N is responding slowly, what could be wrong?", "",
     "Diagnostic checklist: 1) VRAM pressure — check if the sim or other applications are consuming GPU memory. If VRAM is low, I drop to the 3B model or CPU, both slower. 2) Ollama model not loaded — first query after idle has a cold-start penalty while the model loads into VRAM. 3) RAG queries — if the knowledge base is large, retrieval adds latency. 4) CPU competition — check Task Manager for background processes consuming CPU. 5) Thermal throttling — if GPU or CPU is overheating, performance degrades. Check temperatures in the diagnostics panel. Most common cause: VRAM contention. Close unnecessary GPU-using applications."),

    # Physics reasoning
    ("Why is a rear-engine car harder to control in oversteer?", "",
     "Weight distribution and inertia. A rear-engine car has mass concentrated behind the rear axle. When the rear breaks loose: 1) The mass behind the pivot point (rear axle) acts like a pendulum — once it starts rotating, angular momentum carries it further. 2) Recovery requires countersteering AND throttle management — but the weight over the rear means throttle input drastically changes rear grip. 3) The snap is faster because the moment of inertia about the yaw axis is higher at the extremes. Compare to a front-engine car where mass is between the axles — it self-centers more naturally. This is why Porsche 911s (rear-engine) have a reputation for snap oversteer, and why they require precise, progressive driving inputs."),

    # Systems thinking
    ("How do all the car systems interact during a corner?", "",
     "Corner entry to exit — system interactions: BRAKING: brake pressure creates forward weight transfer, loading fronts, unloading rears. ABS modulates to prevent lockup. Brake bias determines which axle reaches limit first. TURN-IN: steering input generates lateral force. Weight transfer (from braking) loads the front, creating grip. Differential behavior determines inside wheel behavior. APEX: peak lateral G. Aero downforce proportional to speed squared adds to mechanical grip. Tire slip angles at maximum — all four tires near their grip limit simultaneously in a balanced car. EXIT: throttle application transfers weight rearward. Differential locks (LSD setting) determine power delivery. Traction control limits wheelspin. As speed increases, aero grip grows, allowing more throttle. Every system affects every other — changing one parameter cascades through the entire dynamic envelope."),

    # Structured problem solving
    ("Help me systematically improve my lap time", "",
     "Improvement framework, ordered by time gained per effort: 1) BRAKING (biggest gains): Find the latest possible braking point for each corner. Threshold brake — max pressure initially, trail-brake into the corner to maintain front load. Most drivers brake too early. 2) CORNER EXIT (second biggest): prioritize exit speed over entry speed, especially before long straights. The time gained on the straight compounds. Get the car straight, then full throttle. 3) RACING LINE: use all the track. Entry: outside. Apex: inside. Exit: outside. Maximize radius = maximize speed. 4) CONSISTENCY: eliminate mistakes before chasing tenths. A clean lap beats a fast-but-messy lap over race distance. 5) SETUP: only after your driving is consistent. Small aero and suspension tweaks for balance. 6) DATA: compare your telemetry to faster drivers. Focus on where the biggest delta is — that's your opportunity."),
]
n = add_batch("e3n-reasoning", reason)
print(f"  General reasoning: {n}/{len(reason)} examples")

# ════════════════════════════════════════════════════════════════
# 7. MECHANICAL ENGINEERING FUNDAMENTALS (for MechE context)
# ════════════════════════════════════════════════════════════════
print("Building MechE fundamentals dataset...")
create_dataset("e3n-meche")
meche = [
    ("Explain stress vs strain", "",
     "Stress (sigma) = Force / Area [Pa or N/m2]. It's the internal resistance per unit area. Strain (epsilon) = change in length / original length [dimensionless]. It's the deformation ratio. Relationship: in the elastic region, stress = E x strain (Hooke's Law), where E is Young's modulus. Key concepts: yield stress is where permanent deformation begins, ultimate stress is the maximum, and fracture stress is where it breaks. In racing: chassis members are designed to operate well below yield stress under normal loads, with safety factors for impact loads."),
    ("What is a moment of inertia and why does it matter?", "",
     "Moment of inertia (I) is the rotational equivalent of mass — it measures resistance to angular acceleration. I = sum(m x r2) where r is the distance from the rotation axis. For a car: yaw moment of inertia determines how quickly it changes direction. Low yaw MOI (mass concentrated near center): quick, responsive turn-in but potentially twitchy. High yaw MOI (mass at extremes, like front engine + rear fuel tank): sluggish turn-in but more stable, predictable. This is why mid-engine cars feel more agile — mass is centralized, minimizing yaw MOI."),
    ("Explain heat transfer modes relevant to a race car", "",
     "Three modes: 1) CONDUCTION: heat transfer through solid contact. Brake pad to rotor, tire to track surface. Rate depends on material conductivity and contact area. Carbon brakes have different thermal conductivity than steel. 2) CONVECTION: heat transfer via fluid flow. Radiator cooling (coolant to air), brake cooling (air through ducts to rotors), engine oil cooling. Forced convection (airflow at speed) is far more effective than natural. 3) RADIATION: electromagnetic heat emission. Hot brakes glow red — they're radiating infrared. Significant at high temps (proportional to T4). In racing: convective cooling is the primary thermal management tool — duct sizing, airflow routing, and radiator capacity are critical design parameters."),
    ("What's fatigue failure and how does it affect race car components?", "",
     "Fatigue failure occurs when cyclic loading causes crack initiation and propagation, even at stress levels below the yield strength. S-N curve: plots stress amplitude vs number of cycles to failure. Each load cycle (suspension compression, brake application, engine firing) accumulates damage. Critical in racing: suspension components experience millions of cycles per race. Wheel bearings, connecting rods, and suspension links are fatigue-critical. Endurance racing amplifies this — 24 hours means 10-100x more cycles than a sprint race. Design approach: calculate expected load spectrum, apply appropriate safety factors, inspect for crack propagation at service intervals, and replace components before calculated fatigue life is reached."),
    ("Explain how a CVT works compared to a traditional gearbox", "",
     "Traditional gearbox: fixed gear ratios via meshed gear pairs. Each gear is a specific ratio — 1st might be 3.5:1, 2nd 2.5:1, etc. Discrete steps mean the engine operates at different points on the power curve in each gear. CVT (Continuously Variable Transmission): uses a belt/chain between two variable-diameter pulleys. By changing pulley widths, the ratio sweeps continuously from low to high. This keeps the engine at peak power RPM at all times — theoretically maximum performance. In practice: CVTs have power handling limitations and efficiency losses in the belt. Racing mostly uses sequential gearboxes with paddle shift for fast, reliable ratio changes. Some series (like Formula Student) have experimented with CVTs for efficiency."),
    ("What's the difference between laminar and turbulent flow?", "",
     "Laminar flow: fluid moves in parallel layers with no mixing. Smooth, predictable, low friction. Occurs at low Reynolds numbers (Re < ~2300 for pipe flow). Turbulent flow: chaotic, mixing, eddies. Higher friction but better heat transfer. Occurs at high Re (> ~4000). In racing aero: 1) Laminar flow over body surfaces reduces drag but is fragile — easily trips to turbulent. 2) Turbulent boundary layer is more resistant to separation — stays attached longer on curved surfaces. 3) Vortex generators deliberately trip laminar to turbulent to prevent separation in critical areas. 4) In the diffuser, managing the boundary layer transition is critical for maintaining attached flow and maximum downforce."),
]
n = add_batch("e3n-meche", meche)
print(f"  MechE fundamentals: {n}/{len(meche)} examples")

# ════════════════════════════════════════════════════════════════
# 8. SIMULATION SCENARIOS — contextual reasoning under pressure
# ════════════════════════════════════════════════════════════════
print("Building simulation scenarios dataset...")
create_dataset("e3n-simulations")
sims = [
    # Multi-factor decision making
    ("Race situation: P2, 5 laps to go, tires are gone, car behind is 1.2 seconds and closing 0.3s per lap. What do I do?", "",
     "Math: 0.3s/lap x 5 laps = 1.5s closure. They'll catch you in 4 laps. Defense strategy: 1) Protect the inside on braking zones — they need to pass, force them to the outside. 2) Use track knowledge — position the car to make passing difficult. Late apex key corners so they can't get alongside on exit. 3) Don't fight too hard in fast corners — a collision costs more than P2 vs P3. 4) Consider where you CAN'T be passed — if there's a section with no overtaking opportunity, save your tires through there. 5) Manage the gap — you don't need to be faster, just unpredictable and well-positioned. 4 laps of smart defense to hold P2."),

    # Emergency decision
    ("Engine warning light just came on! Oil pressure dropping! Lap 47 of 50.", "",
     "CRITICAL. Oil pressure loss is engine-ending. Immediate actions: 1) Reduce RPM immediately — short shift, stay below 6000 RPM. 2) Monitor oil pressure gauge — if it's dropping below minimum (usually 2.0 bar), you're risking seizure. 3) Three laps to go — calculate: can you limp at 80% pace and still finish in points? 4) If oil pressure stabilizes at reduced RPM, nurse it home. If it continues dropping, pit and retire — a blown engine costs far more than the points. 5) Stay off high-RPM corners and avoid aggressive downshifts that spike revs. Priority: preserve the engine for the next race if you can't safely finish this one."),

    # Weather transition
    ("It's starting to drizzle. Track conditions are changing rapidly. I'm on slicks, P6, mid-pack. Half the field hasn't pitted yet.", "",
     "Classic weather decision point. Analysis: 1) P6 mid-pack — you have less to lose than the leaders. 2) Half haven't pitted — if you pit early for inters and rain intensifies, you'll pass everyone who stays out. 3) If rain stops, you'll lose time on inters on a drying track. Risk-reward: at P6, the upside of getting the call right (potentially P2-P3) outweighs the downside (dropping to P8-P9 on inters in the dry). Recommendation: PIT NOW for intermediates. Early changers in rain almost always benefit. The cars that stay out will either pit later (losing track position in chaos) or struggle on slicks in the wet. Be bold from P6 — you're in a position to gain, not defend."),

    # Strategy with incomplete data
    ("Fuel light came on but I don't know exactly how much is left. 8 laps to go.", "",
     "Fuel light typically indicates 2-5 laps of fuel remaining depending on the car. Conservative approach: 1) Immediately switch to max fuel save — lift-and-coast on every straight, short shift to lower RPMs. 2) This typically saves 15-25% fuel per lap. 3) At 8 laps remaining, you need aggressive saving. 4) Draft behind other cars when possible — reduced drag means less fuel consumption. 5) Prioritize finishing over position — running out on the last lap gives you nothing. 6) If you have a comfortable gap behind you, slow down significantly. Better to finish P8 than run out and get P-DNF. Absolute priority: cross the finish line with the engine running."),

    # Multi-car strategy
    ("Three-way battle for P3. I'm the middle car. What's the play?", "",
     "P3 with pressure from behind and an opportunity ahead. Strategy: 1) Attack forward first — the car ahead is your target. Force them to defend, which makes them slower and potentially opens a gap to the car behind. 2) Don't get so focused on attack that the car behind passes you — check mirrors every 2-3 corners. 3) Use the car ahead as a brake reference — if you can brake later than them, you have a passing opportunity. 4) DRS/slipstream: position yourself to use the car ahead's draft, then attack at the end of the straight. 5) If the car behind is faster than both of you, consider letting them pass and using their draft to attack the car ahead. Sometimes losing P4 temporarily gains P3 in the next sequence. Think chess, not checkers."),

    # Technical problem mid-race
    ("Brakes are feeling spongy after 30 laps. What's happening?", "",
     "Brake fade diagnosis: 1) Heat buildup — 30 laps of heavy braking has likely brought rotor temps past optimal. Carbon brakes fade differently than steel. 2) Pad material depletion — less friction material means less bite. 3) Fluid temperature — brake fluid absorbs moisture and can boil at high temps, creating gas bubbles (the spongy feeling). 4) Rotor glazing — sustained high temps can glaze the rotor surface, reducing friction coefficient. Countermeasures NOW: 1) Brake earlier and lighter — extend braking zones. 2) Engine braking — downshift more aggressively to reduce brake load. 3) Cool the brakes on straights — weave slightly to increase airflow through ducts. 4) If performance continues dropping, pit for brake adjustment or fresh pads if possible."),

    # Post-race analysis with context
    ("We finished P4. Started P7. Debrief the race.", "Pitstop 1: lap 18, 28s. Pitstop 2: lap 42, 26s. Best lap: 1:41.8 on lap 35. Worst lap: 1:44.1 on lap 1. No incidents.",
     "Race debrief: Result: P4 from P7 — gained 3 positions, solid performance. Pace: Best 1:41.8 on lap 35 (mid-second stint) — suggests the car was well-suited to the medium fuel load and tire window at that point. Worst 1:44.1 on lap 1 — 2.3s off best pace, typical for cold tires and full fuel on the opening lap. Pit stops: Stop 1 at 28s — slightly slow, investigate what caused the 2s overshoot. Stop 2 at 26s — clean. Strategy: 18+24 lap stints — reasonable distribution. The second stint was longer but produced the best lap, suggesting good tire management. Gains: positions were likely gained through pit stop windows and consistent pace rather than on-track overtakes. Areas for improvement: 1) Opening lap pace — work on tire warm-up technique. 2) Pit stop 1 lost 2 seconds — worth investigating. 3) Could the best lap pace be sustained longer with different tire management? Overall: clean, consistent, well-executed race. P4 with car that qualified P7 is a strong result."),
]
n = add_batch("e3n-simulations", sims)
print(f"  Simulations: {n}/{len(sims)} examples")

# ════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("DATASET BUILD COMPLETE")
r = client.get("/training/datasets")
datasets = r.json().get("datasets", [])
total = 0
for d in datasets:
    count = d.get("count", d.get("examples", 0))
    total += count
    print(f"  {d['name']}: {count} examples")
print(f"\n  TOTAL: {total} training examples across {len(datasets)} datasets")
print("="*60)
