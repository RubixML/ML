<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
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

    #[Test]
    public function transform() : void
    {
        $dataset = new Unlabeled([
            [true, 'false', '1', 1, 45.5],
            [false, '', '0', 0, 0.0],
        ]);

        $dataset->apply($this->transformer);

        $expected = [
            ['!true!', '!true!', '!true!', '!true!', '!true!'],
            ['!false!', '!false!', '!false!', '!false!', '!false!'],
        ];

        $this->assertEquals($expected, $dataset->samples());
    }
}
