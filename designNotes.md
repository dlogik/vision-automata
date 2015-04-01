Ideas for Vision Automata project
=================================

- How does the object look relative to the environment? eg. Will object be contrasted against a green/blue screen. A bright blob on dull background can very well be tracked with color as a feature.

- How much motion is expected? Will the object change in scale and/or rotate? Template matching is not scale invariant, but SURF/SIFT is.

- How fast does it need to run, what's is the quality vs speed trade-off ???

#### Select feature descriptors to use

- Decide whether to use one or a variety of feature descriptors.

- Maybe try different individual or a combination of detectors if speed is not a concern.

- Allow the user to select up to 3 objects in frame 1 of the video, then detect and track starting from frame 2.

- A global find method is required which searches the whole frame, possibly from more than one scale when starting up or after object disappears.
- Another method searches within a smaller region while tracking.
- A gating method is required which indicates whether a find was a successful.
