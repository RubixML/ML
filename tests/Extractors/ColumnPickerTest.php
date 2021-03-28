<?php

namespace Rubix\ML\Tests\Extractors;

use Rubix\ML\Extractors\CSV;
use Rubix\ML\Extractors\Extractor;
use Rubix\ML\Extractors\ColumnPicker;
use PHPUnit\Framework\TestCase;
use IteratorAggregate;
use Traversable;

/**
 * @group Extractors
 * @covers \Rubix\ML\Extractors\ColumnPicker
 */
class ColumnPickerTest extends TestCase
{
    /**
     * @var \Rubix\ML\Extractors\ColumnPicker;
     */
    protected $extractor;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->extractor = new ColumnPicker(new CSV('tests/test.csv', true), [
            'attitude', 'texture', 'class', 'rating',
        ]);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(ColumnPicker::class, $this->extractor);
        $this->assertInstanceOf(Extractor::class, $this->extractor);
        $this->assertInstanceOf(IteratorAggregate::class, $this->extractor);
        $this->assertInstanceOf(Traversable::class, $this->extractor);
    }

    /**
     * @test
     */
    public function extract() : void
    {
        $expected = [
            ['attitude' => 'nice', 'texture' => 'furry', 'class' => 'not monster', 'rating' => '4'],
            ['attitude' => 'mean', 'texture' => 'furry', 'class' => 'monster', 'rating' => '-1.5'],
            ['attitude' => 'nice', 'texture' => 'rough', 'class' => 'not monster', 'rating' => '2.6'],
            ['attitude' => 'mean', 'texture' => 'rough', 'class' => 'monster', 'rating' => '-1'],
            ['attitude' => 'nice', 'texture' => 'rough', 'class' => 'not monster', 'rating' => '2.9'],
            ['attitude' => 'nice', 'texture' => 'furry', 'class' => 'not monster', 'rating' => '-5'],
        ];

        $records = iterator_to_array($this->extractor, false);

        $this->assertEquals($expected, $records);
    }
}
