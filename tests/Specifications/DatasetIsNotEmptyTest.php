<?php

namespace Rubix\ML\Tests\Specifications;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;
use Generator;

/**
 * @group Specifications
 * @covers \Rubix\ML\Specifications\DatasetIsNotEmpty
 */
class DatasetIsNotEmptyTest extends TestCase
{
    /**
     * @test
     * @dataProvider checkProvider
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param bool $valid
     */
    public function check(Dataset $dataset, bool $valid) : void
    {
        if (!$valid) {
            $this->expectException(InvalidArgumentException::class);
        }

        DatasetIsNotEmpty::check($dataset);

        $this->assertTrue($valid);
    }

    /**
     * @return \Generator<array>
     */
    public function checkProvider() : Generator
    {
        yield [
            Unlabeled::quick([
                ['swamp', 'island', 'black knight', 'counter spell'],
            ]),
            true,
        ];

        yield [
            Unlabeled::quick(),
            false,
        ];
    }
}
