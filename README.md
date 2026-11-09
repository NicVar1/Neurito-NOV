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

# Neurito 2.0: aprende a saltar

Para los experimentos dejé el juego corriendo en modo automático 6000 fotogramas (unos 3 minutos a 30 fps, salen como 49 obstáculos) y conté los esquivados y los choques.

## Experimentos

### 1. Prueba el juego sin entrenar. Anota colisiones y obstáculos esquivados.

Sin entrenar Neurito no esquivó ninguno y chocó 39 veces. La red da una probabilidad de salto de más o menos 0.44 o 0.47 para todo, entonces nunca pasa de 0.5 y nunca salta. La precisión en prueba da 44.9%, que es como decir siempre "esperar".

### 2. Entrena de cien en cien épocas; compara la pérdida, la precisión y el comportamiento.

| Épocas | Pérdida | Precisión | Esquivados | Choques |
|---|---|---|---|---|
| 0 | - | 44.9% | 0 | 39 |
| 100 | 0.2601 | 91.2% | 45 | 3 |
| 200 | 0.2306 | 91.9% | 48 | 1 |
| 300 | 0.2066 | 92.3% | 49 | 0 |
| 500 | 0.1368 | 94.1% | 49 | 0 |
| 700 | 0.0858 | 96.5% | 49 | 0 |
| 1000 | 0.0574 | 97.8% | 49 | 0 |

Con solo 100 épocas ya esquiva casi todos y desde 300 ya no choca. Después de eso la pérdida sigue bajando y la precisión sube hasta 97.8%, pero en el juego ya no se nota porque igual esquiva todo. O sea mejorar la precisión de 92% a 98% no cambió el comportamiento.

### 3. Cambia ocultas=12 por ocultas=3 o ocultas=24; compara generalización.

| Ocultas | Épocas | Precisión entrenamiento | Precisión prueba | Choques |
|---|---|---|---|---|
| 3 | 300 | 92.7% | 92.4% | 0 |
| 3 | 1000 | 93.7% | 93.2% | 0 |
| 12 | 300 | 92.7% | 92.3% | 0 |
| 12 | 1000 | 98.0% | 97.8% | 0 |
| 24 | 300 | 93.8% | 93.7% | 1 |
| 24 | 1000 | 98.3% | 97.7% | 0 |

En todos la precisión de entrenamiento y la de prueba quedan casi iguales, entonces generalizan bien y no hay sobreajuste. Con 3 neuronas se queda en 93% porque no le alcanza para marcar bien los dos bordes de la ventana. Con 24 aprende un poco más rápido al inicio pero al final queda igual que con 12. En el juego los tres terminan sin chocar.

### 4. Modifica la tasa de aprendizaje 0.8 a 0.01 o 1.2 y estudia la convergencia.

| Tasa | Pérdida ép. 1 | Pérdida ép. 100 | Pérdida ép. 1000 | Veces que subió la pérdida | Precisión | Choques |
|---|---|---|---|---|---|---|
| 0.01 | 0.7099 | 0.6449 | 0.4302 | 0 | 86.5% | 37 |
| 0.8 | 0.7099 | 0.2601 | 0.0574 | 0 | 97.8% | 0 |
| 1.2 | 0.7099 | 0.2560 | 0.0392 | 292 | 98.3% | 0 |

Con 0.01 converge muy lento, en 1000 épocas sigue en 0.43. Lo curioso es que tiene 86.5% de precisión y aun así choca 37 veces, porque solo salta cuando el obstáculo ya está muy cerca y ya es tarde. Con 0.8 baja estable. Con 1.2 baja más rápido y termina un poco mejor, pero la pérdida subió en 292 de las 1000 épocas, o sea va oscilando. Funciona pero ya está cerca de volverse inestable.

### 5. Cambia la ventana de entrenamiento en etiqueta() y en la generación de positivos. Comprueba si la política resultante es compatible con la física.

Primero revisé con la física del juego en qué momento sirve saltar (tiempo hasta el obstáculo en fotogramas):

| Velocidad | Salto exitoso si el tiempo está entre |
|---|---|
| 4.5 | 4 y 14.5 |
| 5.5 | 4 y 18 |
| 6.5 | 4 y 20 |
| 7.5 | 4 y 21.5 |
| 8.5 | 4 y 22.5 |

Si salta antes de 4 no alcanza a subir, y si salta muy temprano cae antes de pasar el obstáculo (eso pasa más con velocidades bajas).

Luego entrené 1000 épocas con diferentes ventanas:

| Ventana | Precisión | Esquivados | Choques |
|---|---|---|---|
| 7 - 13 (original) | 97.8% | 49 | 0 |
| 2 - 5 | 97.3% | 49 | 0 |
| 4 - 8 | 96.3% | 49 | 0 |
| 10 - 16 | 97.8% | 43 | 5 |
| 15 - 22 | 98.5% | 0 | 37 |
| 25 - 35 | 99.0% | 0 | 39 |

La ventana original es compatible porque está dentro de lo que permite la física para todas las velocidades. La de 10 a 16 a veces choca porque a velocidad baja saltar en 15 o 16 ya es muy temprano. Las de 15-22 y 25-35 chocan siempre porque salta antes de tiempo y cae encima del obstáculo, aunque tienen la precisión más alta. La de 2 a 5 me sorprendió que no chocara, pero es porque Neurito salta en el primer fotograma que entra a la ventana, o sea en 5, que sí funciona. Lo que importa es sobre todo el borde de arriba de la ventana.

## Pregunta central: ¿por qué una red puede clasificar ejemplos con buena precisión y aun así chocar?

Porque la precisión solo mide si la red copia bien la regla de las etiquetas, no si esa regla sirve en el juego. Si la ventana está mal escogida para la física (como 25-35) la red puede tener 99% de precisión y chocar siempre, porque aprendió muy bien algo que está mal. También importa en qué ejemplos se equivoca: en el juego solo importa el momento justo en que tiene que saltar, y un error ahí es un choque aunque el resto de ejemplos estén bien. Eso pasó con la tasa 0.01, que tenía 86.5% pero saltaba tarde. Además la precisión se mide con datos sintéticos que se reparten distinto a lo que pasa en el juego, donde la red decide fotograma por fotograma y solo cuenta la primera vez que pasa de 0.5.

# Neurito 3.0: aprende a encontrar comida en un laberinto

Para los experimentos usé los 716 pares de inicio y meta de prueba (los que no se usaron para entrenar) y para cada uno dejé que Neurito intentara llegar con el mismo límite del juego, 45 intentos. Conté cuántas veces llega a la comida, cuántas choca con paredes y cuántos pasos usa.

## Experimentos

### 1. Observa Neurito sin entrenar. ¿Por qué se atasca?

Sin entrenar en red pura solo llegó en el 3.1% de los casos y chocó con paredes 36 veces en promedio por caso. La red sin entrenar da probabilidades casi iguales (la más alta en promedio es 0.30) y casi siempre elige arriba o abajo. El problema es que cuando choca contra una pared no se mueve, entonces la entrada a la red (mapa, posición y meta) es exactamente la misma y la red vuelve a elegir lo mismo. Se queda chocando con la misma pared hasta que se acaban los 45 intentos. De los fallos, 589 fueron así contra una pared y 105 fueron dando vueltas entre casillas.

### 2. Entrena 100, 200 y 400 épocas. Compara pérdida, precisión y movimientos.

| Épocas | Pérdida | Precisión entrenamiento | Precisión prueba | Llega (red pura) | Paredes por caso | Pasos cuando llega |
|---|---|---|---|---|---|---|
| 0 | - | 25.3% | 27.7% | 3.1% | 36.37 | 1.82 |
| 100 | 0.1352 | 94.1% | 84.9% | 51.3% | 0.91 | 5.54 |
| 200 | 0.0376 | 99.9% | 90.1% | 81.3% | 0.31 | 6.01 |
| 400 | 0.0107 | 100% | 91.5% | 82.7% | 0.19 | 6.04 |

Entre 100 y 200 épocas es donde más mejora, pasa de llegar la mitad de las veces a 81%. De 200 a 400 la pérdida sigue bajando y en entrenamiento ya tiene 100%, pero en prueba solo sube a 91.5% y llega casi igual (82.7%). Cuando llega usa casi el camino más corto (6 pasos, solo 0.04 más que el óptimo). Lo que pasa es que con 91.5% de precisión igual se equivoca en algún paso del camino, y como un camino tiene varios pasos basta un error para que se salga y empiece a dar vueltas. Por eso llega menos veces de lo que dice la precisión. Con 400 épocas casi todos los fallos (121) son por ciclos y solo 3 por paredes.

### 3. Compara red pura frente a navegación asistida. ¿Qué parte corresponde al modelo y cuál a las reglas?

| Épocas | Llega red pura | Llega asistido | Pasos de más en asistido |
|---|---|---|---|
| 0 | 3.1% | 87.8% | 11.52 |
| 100 | 51.3% | 100% | 2.12 |
| 200 | 81.3% | 100% | 0.74 |
| 400 | 82.7% | 100% | 0.56 |

Lo más llamativo es que el modo asistido llega el 87.8% de las veces sin entrenar nada. Eso muestra que las reglas solas (no ir contra paredes y castigar las casillas ya visitadas) ya sirven para explorar el laberinto casi como al azar hasta encontrar la comida, pero dando muchas vueltas (11.5 pasos de más). Lo que aporta el modelo es la dirección hacia la comida: con la red entrenada el asistido llega siempre y casi por el camino más corto (0.56 pasos de más). Entonces las reglas evitan los choques y los ciclos, y la red es la que sabe hacia dónde ir. El resultado de red pura es lo que de verdad aprendió el modelo.

### 4. Cambia las 96 neuronas ocultas por 16, 32 o 128.

| Ocultas | Pérdida | Precisión entrenamiento | Precisión prueba | Llega red pura | Llega asistido |
|---|---|---|---|---|---|
| 16 | 0.0342 | 99.4% | 87.2% | 74.3% | 100% |
| 32 | 0.0161 | 100% | 89.2% | 78.4% | 100% |
| 96 | 0.0107 | 100% | 91.5% | 82.7% | 100% |
| 128 | 0.0103 | 100% | 90.8% | 85.3% | 100% |

(400 épocas en todos)

Todos casi memorizan el entrenamiento, pero con menos neuronas generalizan peor a los pares que no vieron. Con 16 baja a 87.2% en prueba y llega 74.3%. De 96 a 128 ya casi no cambia, la precisión hasta baja un poquito aunque llega un poco más. En asistido no se nota la diferencia porque las reglas tapan los errores.

### 5. Cambia la tasa de aprendizaje 0.45 y observa si converge.

| Tasa | Pérdida ép. 1 | Pérdida ép. 100 | Pérdida ép. 400 | Veces que subió la pérdida | Precisión prueba | Llega red pura |
|---|---|---|---|---|---|---|
| 0.01 | 1.3943 | 1.2474 | 0.6818 | 0 | 73.9% | 20.4% |
| 0.1 | 1.3884 | 0.4438 | 0.1490 | 22 | 87.2% | 60.2% |
| 0.45 | 1.3695 | 0.1352 | 0.0107 | 85 | 91.5% | 82.7% |
| 1.0 | 1.3423 | 0.0270 | 0.0028 | 75 | 93.3% | 85.5% |
| 2.0 | 1.3116 | 0.0072 | 0.0011 | 20 | 93.3% | 85.6% |
| 4.0 | 1.4149 | 0.3255 | 0.4660 | 199 | 79.7% | 38.0% |

Con 0.01 converge pero demasiado lento, a las 400 épocas sigue en 0.68. Con 0.45, 1 y 2 sí converge. Aquí la pérdida sube a veces porque se entrena con mini-lotes, pero en general baja bien. Me sorprendió que 1 y 2 funcionaran mejor que 0.45 en este caso. Con 4 ya no converge, la pérdida subió en 199 de 400 épocas y al final quedó más alta que en la época 100.

### 6. Cambia de mapa.

| Mapa | Casos de prueba | Precisión prueba | Llega red pura | Llega asistido |
|---|---|---|---|---|
| 1 | 210 | 92.4% | 85.7% | 100% |
| 2 | 251 | 90.0% | 78.5% | 100% |
| 3 | 255 | 92.2% | 84.3% | 100% |

Los tres funcionan parecido porque la red entrenó con los tres mapas, solo que no con esos pares de inicio y meta. El mapa 2 es el más difícil, tiene más paredes sueltas en el medio y más caminos parecidos, y ahí la red se equivoca más. Esto no prueba que funcione en un laberinto nuevo, porque la red no ve las paredes, solo sabe el número del mapa.

### 7. Cambiar BFS por DFS y A*

Hice los profesores con DFS y A* (A* con distancia Manhattan) y entrené la red igual, 400 épocas.

| Profesor | Etiquetas iguales a BFS | Camino del profesor es el más corto | Precisión prueba | Llega red pura | Llega asistido | Pasos de más (asistido) |
|---|---|---|---|---|---|---|
| BFS | 100% | siempre | 91.5% | 82.7% | 100% | 0.56 |
| DFS | 50.2% | 7.7% de los casos | 99.9% | 7.5% | 44.8% | 13.30 |
| A* | 96.2% | siempre | 92.6% | 86.3% | 100% | 0.44 |

Con A* Neurito queda igual o un poquito mejor. A* también encuentra el camino más corto, la única diferencia con BFS es cómo desempata cuando hay dos caminos igual de cortos (por eso el 3.8% de etiquetas distintas), y parece que sus desempates son más fáciles de aprender.

Con DFS empeora muchísimo. DFS no busca el camino más corto, toma el primero que encuentra explorando siempre en el mismo orden (arriba primero). Casi el 55% de sus etiquetas son "arriba". Lo raro es que la red tiene 99.9% de precisión en prueba, aprende DFS casi perfecto, pero solo llega el 7.5% de las veces. Es porque si sigues DFS paso a paso recalculando desde cada casilla, el mismo DFS se queda dando vueltas: solo llegó en el 7.7% de los casos. O sea la red copió muy bien a un mal profesor. Esto es otro ejemplo de que la precisión alta no significa que resuelva el problema, depende de qué tan buenas son las etiquetas.
