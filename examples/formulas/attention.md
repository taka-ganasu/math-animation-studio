# Attention

```bash
math-anim plan \
  --formula "Attention(Q, K, V) = softmax\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V" \
  --goal "Attentionが何をしているか直感的に理解したい" \
  --output-dir outputs/attention_plan \
  --no-llm
```
