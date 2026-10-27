# Laptop demo transcript

Elicitor's terminal demo (`elicitor demo`) with the fine-tuned 1.7B student as the belief updater,
recorded 2026-10-27. The question selector is `eig` with the student's own measured ρ, questions
use template wording, and no other model is involved: the person answering replaces the
simulated user.

**Setup.** Windows 10, RTX 3060 Laptop GPU (6 GB), Ollama 0.24.0, the same Q8_0 GGUF
(1.83 GB) and Modelfile that were evaluated on the cluster, created with `ollama create` and run
with [`experiments/demo_student.toml`](../experiments/demo_student.toml). The model used about
3.9 GB of GPU memory. Eight updates took 1.2–2.0 s each (11.9 s for the whole session).

**Same model, same numbers.** Before the demo, `distill eval-offline` on the 516 validation samples
gave agreement 0.973, MAE 0.355, KL 0.074 on the laptop, against 0.972 / 0.361 / 0.075 on the
cluster (SPEC §7.1), with no failed call.

**Ollama version matters.** The first attempt used Ollama 0.40.11, the current release. There the
same model reached only 0.936 agreement (KL 0.137): with a JSON schema as `format`, that version
writes the keys of a nested object in alphabetical order (`{"long": …, "short": …, "week": …,
"weekend": …}`) instead of the schema's order. The student learned to score the values in the
schema's order and mis-scores when its first key is a different value. Without the schema, both
versions give the teacher's reply on the samples checked. The cluster ran 0.24, so the reported
results are unaffected; any model served through Elicitor with a schema, the teachers included,
is exposed on the newer version.

**How the answers were given.** The answers below were written in advance, one per slot, and fed
to Elicitor's `run_demo` (the function behind `elicitor demo`), which prints exactly what the
command prints; the answer is shown after each prompt as a terminal would show it.

```text
Planning a holiday trip for one traveller. Agent: sel=eig,upd=llm,phr=tpl, up to 8 questions.
Answer in your own words; an empty line or 'quit' stops early.

  climate      #####............... 0.25  (undecided)
  setting      ####................ 0.20  (undecided)
  pace         #####............... 0.25  (undecided)
  lodging      ####................ 0.20  (undecided)
  food         #####............... 0.25  (undecided)
  activity     ####................ 0.20  (undecided)
  transport    #####............... 0.25  (undecided)
  budget       #####............... 0.25  (undecided)
  trip_length  #####............... 0.25  (undecided)

[1/8] What kind of surroundings would you like at your destination?
> I'd love to be near the water, falling asleep to the sound of waves.

  climate      #####............... 0.25  (undecided)
> setting      #################... 0.84  beach
  pace         #####............... 0.25  (undecided)
  lodging      ####................ 0.20  (undecided)
  food         #####............... 0.25  (undecided)
  activity     ####................ 0.20  (undecided)
  transport    #####............... 0.25  (undecided)
  budget       #####............... 0.25  (undecided)
  trip_length  #####............... 0.25  (undecided)

  setting:
    beach          #################... 0.84
    city           #................... 0.04
    mountains      #................... 0.04
    countryside    #................... 0.04
    no_preference  #................... 0.04

[2/8] What type of accommodation would you like to stay in?
> I'd rather have my own place with a kitchen than a hotel room.

  climate      #####............... 0.25  (undecided)
  setting      #################... 0.84  beach
  pace         #####............... 0.25  (undecided)
> lodging      #########........... 0.47  rental
  food         #####............... 0.25  (undecided)
  activity     ####................ 0.20  (undecided)
  transport    #####............... 0.25  (undecided)
  budget       #####............... 0.25  (undecided)
  trip_length  #####............... 0.25  (undecided)

  lodging:
    hotel          .................... 0.02
    rental         #########........... 0.47
    hostel         #####............... 0.24
    camping        #####............... 0.24
    no_preference  .................... 0.02

[3/8] What would you most like to spend your time doing?
> Snorkelling for sure, maybe a hike if it isn't too steep.

  climate      #####............... 0.25  (undecided)
  setting      #################... 0.84  beach
  pace         #####............... 0.25  (undecided)
  lodging      #########........... 0.47  rental
  food         #####............... 0.25  (undecided)
> activity     ############........ 0.60  adventure
  transport    #####............... 0.25  (undecided)
  budget       #####............... 0.25  (undecided)
  trip_length  #####............... 0.25  (undecided)

  activity:
    adventure      ############........ 0.60
    culture        #................... 0.03
    nightlife      #................... 0.03
    wellness       #................... 0.03
    no_preference  ######.............. 0.31

[4/8] What kind of weather would you like on this trip?
> Honestly I just want to be warm. Somewhere I can wear shorts every single day.

> climate      ##################.. 0.88  no_preference
  setting      #################... 0.84  beach
  pace         #####............... 0.25  (undecided)
  lodging      #########........... 0.47  rental
  food         #####............... 0.25  (undecided)
  activity     ############........ 0.60  adventure
  transport    #####............... 0.25  (undecided)
  budget       #####............... 0.25  (undecided)
  trip_length  #####............... 0.25  (undecided)

  climate:
    tropical       #................... 0.04
    mild           #................... 0.04
    cold           #................... 0.04
    no_preference  ##################.. 0.88

[5/8] How busy would you like your days to be?
> I want to do nothing for most of the day, maybe one thing in the afternoon.

  climate      ##################.. 0.88  no_preference
  setting      #################... 0.84  beach
> pace         ##################.. 0.88  relaxed
  lodging      #########........... 0.47  rental
  food         #####............... 0.25  (undecided)
  activity     ############........ 0.60  adventure
  transport    #####............... 0.25  (undecided)
  budget       #####............... 0.25  (undecided)
  trip_length  #####............... 0.25  (undecided)

  pace:
    relaxed        ##################.. 0.88
    balanced       #................... 0.04
    packed         #................... 0.04
    no_preference  #................... 0.04

[6/8] What kind of food experience are you hoping for?
> I'm not picky, I eat whatever the locals eat, the cheaper the better.

  climate      ##################.. 0.88  no_preference
  setting      #################... 0.84  beach
  pace         ##################.. 0.88  relaxed
  lodging      #########........... 0.47  rental
> food         ########............ 0.39  (undecided)
  activity     ############........ 0.60  adventure
  transport    #####............... 0.25  (undecided)
  budget       #####............... 0.25  (undecided)
  trip_length  #####............... 0.25  (undecided)

  food:
    street_food    ########............ 0.39
    fine_dining    .................... 0.02
    vegetarian     ####................ 0.20
    no_preference  ########............ 0.39

[7/8] How would you like to get around once you're there?
> I don't really care how we get around.

  climate      ##################.. 0.88  no_preference
  setting      #################... 0.84  beach
  pace         ##################.. 0.88  relaxed
  lodging      #########........... 0.47  rental
  food         ########............ 0.39  (undecided)
  activity     ############........ 0.60  adventure
> transport    ########............ 0.39  no_preference
  budget       #####............... 0.25  (undecided)
  trip_length  #####............... 0.25  (undecided)

  transport:
    transit        ####................ 0.20
    car            ####................ 0.20
    tours          ####................ 0.20
    no_preference  ########............ 0.39

[8/8] Roughly how much would you like to spend per day, not counting flights?
> Around 150 dollars a day, give or take.

  climate      ##################.. 0.88  no_preference
  setting      #################... 0.84  beach
  pace         ##################.. 0.88  relaxed
  lodging      #########........... 0.47  rental
  food         ########............ 0.39  (undecided)
  activity     ############........ 0.60  adventure
  transport    ########............ 0.39  no_preference
> budget       ##################.. 0.88  mid
  trip_length  #####............... 0.25  (undecided)

  budget:
    low     #................... 0.04
    mid     ##################.. 0.88
    high    #................... 0.04
    luxury  #................... 0.04

Top recommendations (expected utility under the final belief):
  1. i013  EU 0.73  climate=cold, setting=beach, pace=relaxed, lodging=camping, food=street_food, activity=adventure, transport=tours, budget=high, trip_length=short
  2. i056  EU 0.71  climate=tropical, setting=beach, pace=relaxed, lodging=hostel, food=vegetarian, activity=adventure, transport=tours, budget=high, trip_length=week
  3. i137  EU 0.71  climate=tropical, setting=beach, pace=relaxed, lodging=hostel, food=street_food, activity=adventure, transport=car, budget=low, trip_length=long
```

**What the session shows.**

- Eight questions in 12 s on a laptop. Every question is an open one, the type the student's ρ
  (0.99) rates most reliable, on the slot with the most uncertainty.
- Free text without the domain's keywords is read correctly: "falling asleep to the sound of
  waves" → beach 0.84; "my own place with a kitchen" → rental, hotel ruled out; "Snorkelling … a
  hike" → adventure; "around 150 dollars a day" → mid.
- Uncertainty is kept where the answer is vague: "whatever the locals eat, the cheaper the better"
  scores street food and no preference both 10, so the two stay level. "I don't really care" moves
  transport only part of the way: no preference gets 10, but the modes get 5 ("no information")
  rather than 0, so one answer only doubles the odds of indifference.
- **One miss.** "I just want to be warm … shorts every single day" was read as *no preference*
  (0.88) for climate. The student's reply quoted the right evidence (`"evidence":"warm"`) but
  gave `no_preference` 10 and the climates 0. The 32B teacher does not do this: of the 120
  training and validation prompts on climate that mention "warm", it put `tropical` on top in 90.
  The miss then shows in the ranking: the top item has a cold climate, which under the final
  belief costs nothing. It is one example of the 2–3 % of updates where the student and the
  teacher disagree (test agreement 0.980).
