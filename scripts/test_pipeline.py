from pathlib import Path
import subprocess
import sys
import polars as pl


IMAGE_NAME = "magnit-uplift"
TEST_INPUT = "data/train.parquet"
OUT_RUN_1 = "artifacts/predictions_stub_1.csv"
OUT_RUN_2 = "artifacts/predictions_stub_2.csv"


def run_cmd(cmd: list, description: str) -> None:
    print(f"\n🚀 {description}...")
    print(f"Running: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print(
            "\n❌ Ошибка: Команда 'docker' не найдена. "
            "Убедитесь, что Docker установлен, запущен и добавлен в переменную окружения PATH.",
            file=sys.stderr,
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Ошибка при выполнении: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(e.returncode)


def validate_results(pred_path: Path, test_path: Path, pred2_path: Path) -> None:
    print("\n🔍 Начинаем валидацию результатов...")

    if not pred_path.exists():
        print(f"❌ FAIL: Файл предсказаний не найден: {pred_path}", file=sys.stderr)
        sys.exit(1)

    df_pred = pl.read_csv(pred_path)

    test_len = pl.scan_parquet(test_path).select(pl.len()).collect().item()

    # Проверка 1: совпадение строк
    if len(df_pred) != test_len:
        print(
            f"❌ FAIL: Несовпадение строк. Ожидалось {test_len}, получено {len(df_pred)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Проверка 2: названия колонок
    expected_cols = {"user_id", "uplift_score"}
    if set(df_pred.columns) != expected_cols:
        print(
            f"❌ FAIL: Ошибка в названиях колонок. Ожидалось {expected_cols}, получено {df_pred.columns}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Проверка 3: пропуски (NaN, Null)
    null_count = df_pred.null_count().sum_horizontal().item()
    if null_count > 0:
        print(
            "❌ FAIL: В файле предсказаний обнаружены NaN/Null значения",
            file=sys.stderr,
        )
        sys.exit(1)

    if not pred2_path.exists():
        print(
            f"❌ FAIL: Второй файл {pred2_path} не найден для проверки детерминизма",
            file=sys.stderr,
        )
        sys.exit(1)

    df_pred2 = pl.read_csv(pred2_path)

    # Проверка 4: отличие результатов (детерминизм)
    if not df_pred.equals(df_pred2):
        print(
            "❌ FAIL: Результаты двух запусков различаются. Алгоритм недетерминирован.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("✅ PASS: Тест на детерминизм пройден")
    print("\n🎉 Все тесты успешно пройдены!")

    try:
        pred_path.unlink(missing_ok=True)
        pred2_path.unlink(missing_ok=True)
        print("🧹 Временные тестовые файлы успешно удалены")
    except Exception as e:
        print(
            f"⚠️ Предупреждение: не удалось удалить временные файлы: {e}",
            file=sys.stderr,
        )


def main():
    project_dir = Path(__file__).parent.parent.resolve()

    data_dir_host = project_dir / "data"
    artifacts_dir_host = project_dir / "artifacts"

    artifacts_dir_host.mkdir(exist_ok=True)

    run_cmd(["docker", "build", "-t", IMAGE_NAME, "."], "Сборка Docker-образа")

    run_cmd(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{data_dir_host.as_posix()}:/app/data",
            "-v",
            f"{artifacts_dir_host.as_posix()}:/app/artifacts",
            IMAGE_NAME,
            "--test",
            f"/app/{TEST_INPUT}",
            "--out",
            f"/app/{OUT_RUN_1}",
            "--stub",
        ],
        "Первый запуск инференса в контейнере",
    )

    run_cmd(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{data_dir_host.as_posix()}:/app/data",
            "-v",
            f"{artifacts_dir_host.as_posix()}:/app/artifacts",
            IMAGE_NAME,
            "--test",
            f"/app/{TEST_INPUT}",
            "--out",
            f"/app/{OUT_RUN_2}",
            "--stub",
        ],
        "Второй запуск инференса (проверка детерминизма)",
    )

    validate_results(
        artifacts_dir_host / "predictions_stub_1.csv",
        project_dir / TEST_INPUT,
        artifacts_dir_host / "predictions_stub_2.csv",
    )


if __name__ == "__main__":
    main()
