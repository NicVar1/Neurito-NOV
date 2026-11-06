# Neurito

## Experimentos

### 1. Mantén dos necesidades fijas y varía la tercera. ¿Cuándo cambia la acción predicha?

Probé dejando energía en 0.5 y aburrimiento en 0.5 y fui subiendo el hambre de 0 a 1. Hasta 0.4 la red dice jugar (jugar 0.46, dormir 0.46, comer 0.07) y en 0.5 ya cambia a comer (0.35). Desde 0.6 comer está en 0.78 y en 0.8 ya casi es 1.

Con hambre 0.5 y aburrimiento 0.3 variando la energía: con energía entre 0 y 0.4 duerme, y en 0.5 cambia a comer (comer 0.499 vs dormir 0.491, casi empate).

Con hambre 0.4 y energía 0.7 variando el aburrimiento: hasta 0.4 come y en 0.5 cambia a jugar (0.85).

O sea la acción cambia cuando la necesidad que estoy moviendo pasa a la mayor de las otras dos, que es justo la regla con la que se hicieron las etiquetas. Cerca de ese punto las probabilidades quedan parecidas y lejos de ahí una sola se va casi a 1.

### 2. Compara prioridades casi iguales. ¿Cómo cambian las salidas de Softmax?

Cuando dos prioridades están casi iguales Softmax reparte la probabilidad entre esas dos y la otra queda en 0. Por ejemplo:

- hambre 0.7 y cansancio 0.72 → comer 0.40, dormir 0.60
- hambre 0.7 y cansancio 0.70 → comer 0.49, dormir 0.51
- hambre 0.7 y cansancio 0.68 → comer 0.58, dormir 0.42
- hambre 0.7 y aburrimiento 0.68 → comer 0.61, jugar 0.39

Si las tres están iguales (por ejemplo 0.5, 0.5, 0.5) sale casi 1/3 para cada una (0.35, 0.33, 0.31). Entonces con una diferencia de 0.02 ya cambia la acción pero la red no está muy segura. En los empates exactos a veces la red no coincide con la etiqueta, pero eso es porque argmax escoge la primera cuando hay empate, entonces ahí no hay una respuesta "correcta" de verdad.

### 3. Cambia la capa oculta de 6 a 3, 10 o 20 neuronas; compara pérdida y precisión.

| Neuronas | Pérdida final | Precisión |
|---|---|---|
| 3 | 0.1618 | 96.3% |
| 6 | 0.1225 | 99.4% |
| 10 | 0.1262 | 99.3% |
| 20 | 0.1220 | 98.8% |

Con 3 neuronas baja como 3% la precisión, le falta capacidad para separar bien las tres acciones. Con 6, 10 y 20 da casi lo mismo, así que para este problema 6 ya alcanza y poner más no ayuda.

### 4. Cambia las épocas o la tasa de aprendizaje. ¿El entrenamiento mejora, se estanca o se vuelve inestable?

Épocas (con tasa 0.1):

| Épocas | Pérdida | Precisión |
|---|---|---|
| 50 | 0.9388 | 60.6% |
| 200 | 0.4984 | 91.8% |
| 600 | 0.1944 | 98.6% |
| 1200 | 0.1225 | 99.4% |
| 3000 | 0.0711 | 99.5% |
| 10000 | 0.0368 | 99.7% |

Con pocas épocas no alcanza a aprender. Después de 1200 la pérdida sigue bajando pero la precisión casi no sube, se estanca.

Tasa de aprendizaje (con 1200 épocas):

| Tasa | Pérdida | Precisión |
|---|---|---|
| 0.001 | 1.0333 | 41.8% |
| 0.01 | 0.7376 | 82.9% |
| 0.1 | 0.1225 | 99.4% |
| 0.5 | 0.0484 | 99.6% |
| 1 | 0.0447 | 98.7% |
| 2 | 0.0533 | 97.5% |
| 5 | 1.0979 | 33.8% |
| 10 | 2.8771 | 33.8% |

Con tasa muy baja aprende demasiado lento. Entre 0.1 y 0.5 mejora bien. Con 1 y 2 la pérdida empieza a subir y bajar (oscila) y con 5 o 10 se vuelve inestable, la pérdida queda peor que al inicio y la red predice siempre lo mismo (33.8% que es como adivinar).

### 5. ¿Qué datos y etiquetas requeriría una cuarta acción?

Por ejemplo si quiero que Neurito se bañe necesitaría una cuarta necesidad, suciedad. Los datos tendrían 4 columnas: hambre, energía, aburrimiento y suciedad. La etiqueta sería el argmax de [hambre, 1 - energía, aburrimiento, suciedad], o sea valores de 0 a 3. También habría que cambiar W1 a (4, 6), W2 a (6, 4) y b2 a (1, 4) para que salgan 4 probabilidades, y agregar la acción, los textos y las imágenes de bañarse. Además hay que revisar que haya suficientes ejemplos de cada acción.

## Explica tus resultados

### ¿En qué se diferencian pesos y sesgos?

Los pesos multiplican las entradas y dicen qué tanto influye cada entrada en cada neurona (y si la sube o la baja). El sesgo se suma después y no depende de la entrada, sirve para mover el punto donde la neurona se activa. En z = x·W + b, W cambia la inclinación y b la desplaza.

### ¿Por qué el cansancio es 1 - energía?

Porque en hambre y aburrimiento un valor alto significa que necesita hacer algo, pero con la energía es al contrario, mucha energía significa que no necesita dormir. Haciendo 1 - energía las tres quedan en el mismo sentido (alto = más urgente) y se pueden comparar directo con argmax. Energía 0.9 da cansancio 0.1 y energía 0.1 da cansancio 0.9.

### ¿Qué hace ReLU y qué hace Softmax?

ReLU está en la capa oculta y hace max(0, x), deja pasar los positivos y los negativos los vuelve 0. Eso le da no linealidad a la red, sin eso las capas serían lo mismo que una sola operación lineal.

Softmax está en la salida y convierte los 3 valores en probabilidades que suman 1. La acción que se escoge es la de mayor probabilidad. Como se vio en los experimentos, cuando una necesidad domina da casi 1 y cuando hay empate reparte.

### ¿Por qué una precisión alta puede ser engañosa?

El 99.4% se mide contra la misma regla que se usó para crear los datos, entonces solo dice que la red copió bien esa regla, no que se comporte bien en cualquier caso. Además la mayoría de ejemplos son fáciles porque una necesidad es claramente mayor. Probé con 20000 ejemplos y cuando la diferencia entre las dos necesidades más altas es menor a 0.02 la precisión baja a 86.4%, mientras que en el resto es casi 100%. También la precisión no muestra qué tan segura está la red (51% cuenta igual que 99%), y si los datos estuvieran desbalanceados se podría tener una precisión alta prediciendo casi siempre la misma acción.
