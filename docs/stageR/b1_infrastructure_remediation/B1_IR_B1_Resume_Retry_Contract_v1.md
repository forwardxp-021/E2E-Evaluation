# B1-IR B1 Resume and Retry Contract v1

B1 resume is not authorized by this document. The planned denominator remains the original 20 SESSION pairs; no pair, arm, identity, session or roster member may be dropped or replaced.

For B1-TSB-01 through B1-TSB-17, both original attempts and trajectories are authoritative. Official metrics were recovered offline, so rerun is forbidden.

For B1-TSB-18, the baseline remains first-attempt NOT_RUN. Treatment attempt 1 is permanently retained as `INFRASTRUCTURE_FAILURE`. A treatment attempt 2 is allowed only after route infrastructure repair and explicit Owner authorization, with the same pair, scenario, TSB, analyzer and thresholds and a new execution-attempt ID. Both attempts remain in the record.

B1-TSB-19 arms remain first-attempt NOT_RUN and route-compatible. B1-TSB-20 arms remain first-attempt NOT_RUN but route-incompatible. They cannot enter runner until an explicit Owner resume authorization; B1-TSB-20 additionally requires route infrastructure closure.

The maximum is two total attempts per arm. Eligible retry classes are infrastructure, callback-finalization, serializer, artifact-persistence and native-route infrastructure failures. Scientific failure, realized-behavior measurement invalidity, LOW_SPEED_ENDSTOP, mechanism failure, F_match failure and official-safety failure are never retry-eligible. A retry never changes the denominator; the latest Owner-authorized technically complete attempt is authoritative while all earlier attempts remain immutable.
