<?php

namespace Rubix\ML\Tests\NeuralNet\NumPower;

use Generator;
use NumPower;
use Tensor\Matrix;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\TestCase;
use function Apphp\PrettyPrint\pp;

#[Group('NumPower')]
class NumPowerTest extends TestCase
{
    public static function determinantCases() : Generator
    {
        yield 'singular matrix' => [
            [
                [1.0, 2.0, 3.0],
                [2.0, 4.0, 6.0],
                [3.0, 6.0, 9.0],
            ],
        ];

        yield '2x2 positive values' => [
            [
                [6.0, 4.0],
                [2.0, 5.0],
            ],
        ];

        yield '3x3 mixed values' => [
            [
                [4.0, 3.0, 2.0],
                [3.0, 2.0, 1.0],
                [2.0, 1.0, 3.0],
            ],
        ];

        yield '4x4 upper triangular' => [
            [
                [3.0, 1.0, 2.0, 4.0],
                [0.0, 5.0, 6.0, 7.0],
                [0.0, 0.0, 8.0, 9.0],
                [0.0, 0.0, 0.0, 10.0],
            ],
        ];
    }

    #[Test]
    #[TestDox('NumPower transpose swaps axes')]
    public function testNumPowerTransposeSwapsAxes() : void
    {
        $rows = [];

        for ($i = 0; $i < 3; ++$i) {
            $row = [];

            for ($j = 0; $j < 256; ++$j) {
                $row[] = (float) ($i * 1000 + $j);
            }

            $rows[] = $row;
        }

        $x = NumPower::array($rows);

        $t = NumPower::transpose($x, [1, 0]);

        self::assertSame([256, 3], $t->shape());

        $a = $t->toArray();

        self::assertEqualsWithDelta(0.0, (float) $a[0][0], 1e-12);
        self::assertEqualsWithDelta(1000.0, (float) $a[0][1], 1e-12);
        self::assertEqualsWithDelta(2000.0, (float) $a[0][2], 1e-12);

        self::assertEqualsWithDelta(255.0, (float) $a[255][0], 1e-12);
        self::assertEqualsWithDelta(1255.0, (float) $a[255][1], 1e-12);
        self::assertEqualsWithDelta(2255.0, (float) $a[255][2], 1e-12);

        self::assertEqualsWithDelta(42.0, (float) $a[42][0], 1e-12);
        self::assertEqualsWithDelta(1042.0, (float) $a[42][1], 1e-12);
        self::assertEqualsWithDelta(2042.0, (float) $a[42][2], 1e-12);
    }

    #[Test]
    #[TestDox('NumPower determinant matches Matrix determinant')]
    #[DataProvider('determinantCases')]
    public function testNumPowerDeterminantMatchesMatrixDeterminant(array $matrix) : void
    {
        $ndArray = NumPower::array($matrix);
        $matrix = Matrix::build($matrix);

        self::assertEqualsWithDelta($matrix->det(), NumPower::det($ndArray), 1e-3);
    }
}
