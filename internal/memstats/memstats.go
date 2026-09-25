/*
Package memstats — сколько памяти занимает процесс прямо сейчас.

Значение нужно не само по себе, а как ответ на вопрос «сколько я стою».
Проект живёт на телефоне и на бесплатном VPS, и перенос подсистемы памяти в Go
имеет смысл ровно настолько, насколько он уменьшает RSS. Поэтому сервис
показывает свой размер в `/v1/stats` — цифра, которую можно сверить с
`ps`-ом, а не обещание из README.

Читается `/proc/self/statm` (Linux/Termux). На других платформах честно
возвращается 0: неизвестный размер лучше выдуманного.
*/
package memstats

import (
	"os"
	"strconv"
	"strings"
)

// RSSBytes — резидентный размер процесса в байтах, либо 0, если его узнать
// не удалось (не Linux, нет прав, изменился формат /proc).
func RSSBytes() int64 {
	raw, err := os.ReadFile("/proc/self/statm")
	if err != nil {
		return 0
	}
	fields := strings.Fields(string(raw))
	if len(fields) < 2 {
		return 0
	}
	pages, err := strconv.ParseInt(fields[1], 10, 64)
	if err != nil {
		return 0
	}
	return pages * int64(os.Getpagesize())
}
