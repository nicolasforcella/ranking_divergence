# Generative Frontiers Reloaded

Objective: Evaluating models requires access to the likelihood of the model. For diffusion models, trained on an ELBO objective, this is not easily accessible.

when training a language model, the final objective is to minimize the $KL(p_{ref}, p_{gen})$.

We can rewrite the $KL(p_{ref}, p_{gen})$ as:

$$
{\mathrm{KL}}(q_{\text{gen}} \,\|\, p_{\text{ref}}) = H(q_{\text{gen}}, p_{\text{ref}}) - H(q_{\text{gen}})
$$

If we assume that a sufficiently well trained model can correctly match the generative distribution, then $ p_{ref} \approx p_{scorer} $. This assumption makes the cross entropy term tractable.

Since we have no access to $ H(q_{\text{gen}}) $,  Generative Frontiers proposed a method for ordering models by matching entropy or GenPPL. With the assumption that unigram entropy would provide a correct estimate, enough to rank the models.

As we can see in the following plot, genertive frontiers does not allow for separating AR models:

![alt text](figures/gen_frontiers_unigram_entropy_ar.png)

It has been also shown that it can be hacked by degenerate samplers.










