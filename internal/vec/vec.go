/*
Package vec — минимальная векторная математика в float64.

Отдельный пакет, а не функции внутри diary: этим же косинусом пользуется
дедупликация выдачи в RAG, и им же будет пользоваться локальный ONNX-движок
(сравнение векторов «свой ли это дубль»). Дублировать формулу в двух местах
опасно: разойдясь в деталях (нормировка, поведение нулевого вектора), она
даст два разных ответа на один и тот же вопрос «похоже или нет».

float64, а не float32 — сознательно: это цена побайтового совпадения с numpy
из Python-версии. Экономить здесь нечего, векторов в памяти немного.
*/
package vec

import "math"

// Dot — скалярное произведение. Паника при разной длине — намеренная: смешать
// векторы разной размерности значит получить тихо неверный результат, а не
// ошибку, поэтому лучше упасть в тесте, чем разойтись с Python в выдаче.
func Dot(left, right []float64) float64 {
	if len(left) != len(right) {
		panic("vec: dot product of vectors with different dimensions")
	}
	sum := 0.0
	for index, value := range left {
		sum += value * right[index]
	}
	return sum
}

// Norm — евклидова длина вектора.
func Norm(vector []float64) float64 { return math.Sqrt(Dot(vector, vector)) }

// Cosine — косинусное сходство; 0, если любой из векторов нулевой (ровно как
// `np.where(denom > 0, ..., 0.0)` в Python-версии).
func Cosine(left, right []float64) float64 {
	denominator := Norm(left) * Norm(right)
	if denominator <= 0 {
		return 0
	}
	return Dot(left, right) / denominator
}
