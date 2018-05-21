#### 1. Motivation 
<!-- all concepts here are very high level, as just an intuition  -->
+ explain that for key exchange you need privacy
<!-- so this is referring to Maurers result ? Don't you need to introduce information theoretic secret key for that? -->
+ Then define that for an eavesdropper it means to factor it out: `P(X,Y,Z) = P(X,Y)P(Z)`
    * possible quotation of Shannon on information theoretical security
<!-- Maybe you find some reference to a source that discusses information theoretic secret key in one of the papers.... -->
+ QM offers an analogy to privacy in QE : correlation + privacy
    * some analogies work
    * some don't
    * some are yet to be shown
#### 2. A common key exchange problem
1. What is a shared key?
    + high level explanation
    + formal definition but with remainders to next sections for concepts of *random variables*, *mutual information*
2. The analogy with entanglement
    + high level explanation of entanglement.
    <!--The complete definition of entanglement is in the appendix -->
    <!-- Can you draw an analogy without information theoretic secret keys... that is without the content of the next chapter? -->
3. Examples of key exchange
    1. Diffie-Hellman
    2. BB84
    3. One-time pad
        + example of utilization of the key after generation
        + info theoretical key exchanged with OTP over already existing channel
        <!-- then I can say that security of new key depends on security of previous channel -->
4. A comparison between securities
    <!-- the short explanation on what it means to be secure on computational level, Physics level or info theoretical level -->
5. The equivalent of CKA in QM
    1. Mixed entangled states
    2. Quantum distillation
    3. A cost to entanglement
<!-- still not sure of content of this part since it's very technical. I know only at high level -->
<!-- Important here is: monogamy of entanglement: if a system kann be shown to be maximally entangled, it cannot be entangled with any other system... it is product, similar to a 3p prob distrib, being product to E (and in fact it can then be shown that one can use if for key agreement (http://cqi.inf.usi.ch/qic/91_Ekert.pdf ) -->
#### 3. Information Theoretical model of cryptography: random variables
1. The abstraction through random variables
    + a message is a repetition of an experiment defined by random variables with a probability distribution (on a channel)
2. Local operations and public communication
    + operations done to the prob distribution
    + retake Diffie-Hellman as practical example (or BB84)
3. Secret key rate
    + how many bit of *secrecy* are extractable from LOPC for a given probability distribution?
#### 4. Measures of correlation and their bounds
1. Mutual information
2. An eavesdropper that can choose the best channel to listen to.
    + define intrinsic information with the minimum of MI over all channels
<!-- here I wanted to jump back to QM, saying that there are known states the equivalent of key cost not 0, but non distillable
... but I couldn't find a valid one. Do you have any ideas? -->
3. When correlation is unusable
    1. Undistilalble entangled states
        + bound entanglement
    2. The analog in information theory
        + bound information
#### 5. State of research
+ tripartite BI has been found (but it's a different thing)
+ the gaps between the bounds can be arbitrarily large
+ a tripartite probability distribution that is asymptotically BI has been found
#### 6. A practical analysis of a candidate distribution
1. analysis design and goals
2. different noises analysis
3. scaling up dimensions
<!-- this part is still a bit explorative because we are still "playing" with the tests -->
#### 7. Conclusion

---

##### Appendix A: Mathematical framework for QM
+ inner product spaces
+ tensor product spaces
+ linear operators
+ self-adjoint operators

##### Appendix B: Quantum Mechanics
+ Dirac's notation
+ measurements on a basis
+ quantum entanglement

##### Appendix C: Information Theory
+ entropy
+ channels
