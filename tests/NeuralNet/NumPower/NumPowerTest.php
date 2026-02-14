<?php

namespace Rubix\ML\Tests\NeuralNet\NumPower;

use NumPower;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\TestCase;

#[Group('NumPower')]
class NumPowerTest extends TestCase
{
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
}
