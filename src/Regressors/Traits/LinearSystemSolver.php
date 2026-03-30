<?php

namespace Rubix\ML\Regressors\Traits;

use Rubix\ML\Exceptions\RuntimeException;

use function abs;
use function count;

trait LinearSystemSolver
{
    /**
     * @param list<list<float|int>> $a
     * @param list<float|int> $b
     * @return list<float>
     */
    private static function solveLinearSystemWithJitter(array $a, array $b) : array
    {
        $jitter = 0.0;

        for ($attempt = 0; $attempt < 6; ++$attempt) {
            try {
                $aTry = $a;

                if ($jitter > 0.0) {
                    $n = count($aTry);

                    for ($i = 0; $i < $n; ++$i) {
                        $aTry[$i][$i] = (float) $aTry[$i][$i] + $jitter;
                    }
                }

                return self::solveLinearSystem($aTry, $b);
            } catch (RuntimeException) {
                $jitter = $jitter > 0.0 ? $jitter * 10.0 : 1.0e-12;
            }
        }

        throw new RuntimeException('Unable to solve linear system (matrix may be singular or ill-conditioned).');
    }

    /**
     * @param list<list<float|int>> $a
     * @param list<float|int> $b
     * @return list<float>
     */
    private static function solveLinearSystem(array $a, array $b) : array
    {
        $n = count($a);

        if ($n < 1 || count($b) !== $n) {
            throw new RuntimeException('Invalid linear system dimensions.');
        }

        for ($i = 0; $i < $n; ++$i) {
            if (!isset($a[$i]) || count($a[$i]) !== $n) {
                throw new RuntimeException('Coefficient matrix must be square.');
            }
        }

        $aug = [];

        for ($i = 0; $i < $n; ++$i) {
            $row = [];

            for ($j = 0; $j < $n; ++$j) {
                $row[] = (float) $a[$i][$j];
            }

            $row[] = (float) $b[$i];
            $aug[] = $row;
        }

        $tol = 1.0e-15;

        for ($col = 0; $col < $n; ++$col) {
            $pivotRow = $col;
            $pivotVal = abs($aug[$col][$col]);

            for ($row = $col + 1; $row < $n; ++$row) {
                $val = abs($aug[$row][$col]);

                if ($val > $pivotVal) {
                    $pivotVal = $val;
                    $pivotRow = $row;
                }
            }

            if ($pivotVal <= $tol) {
                throw new RuntimeException('Singular matrix (pivot too small).');
            }

            if ($pivotRow !== $col) {
                $tmp = $aug[$col];
                $aug[$col] = $aug[$pivotRow];
                $aug[$pivotRow] = $tmp;
            }

            $pivot = $aug[$col][$col];

            for ($j = $col; $j <= $n; ++$j) {
                $aug[$col][$j] /= $pivot;
            }

            for ($row = 0; $row < $n; ++$row) {
                if ($row === $col) {
                    continue;
                }

                $factor = $aug[$row][$col];

                if (abs($factor) <= $tol) {
                    $aug[$row][$col] = 0.0;

                    continue;
                }

                for ($j = $col; $j <= $n; ++$j) {
                    $aug[$row][$j] -= $factor * $aug[$col][$j];
                }

                $aug[$row][$col] = 0.0;
            }
        }

        $x = [];

        for ($i = 0; $i < $n; ++$i) {
            $x[] = (float) $aug[$i][$n];
        }

        return $x;
    }
}
