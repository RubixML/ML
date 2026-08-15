<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\BooleanConverter;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[CoversClass(BooleanConverter::class)]
class BooleanConverterTest extends TestCase
{
    protected BooleanConverter $transformer;

    protected function setUp() : void
    {
        $this->transformer = new BooleanConverter(trueValue: '!true!', falseValue: '!false!');
    }

    public function testTransform() : void
    {
        $dataset = new Unlabeled([
            [true, 'true', '1', 1],
            [false, 'false', '0', 0],
        ]);

        $dataset->apply($this->transformer);

        $expected = [
            ['!true!', 'true', '1', 1],
            ['!false!', 'false', '0', 0],
        ];

        $this->assertEquals($expected, $dataset->samples());
    }
}
