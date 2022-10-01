<?php

namespace Rubix\ML\Tests\Extractors;

use Rubix\ML\Extractors\CSV;
use Rubix\ML\Extractors\Extractor;
use Rubix\ML\Extractors\ColumnFilter;
use PHPUnit\Framework\TestCase;
use IteratorAggregate;
use Traversable;

/**
 * @group Extractors
 * @covers \Rubix\ML\Extractors\ColumnFilter
 */
class ColumnFilterTest extends TestCase
{
    /**
     * @var \Rubix\ML\Extractors\ColumnFilter;
     */
    protected $extractor;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->extractor = new ColumnFilter(new CSV('tests/test.csv', true), [
            'texture',
        ]);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(ColumnFilter::class, $this->extractor);
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
            ['attitude' => 'nice', 'class' => 'not monster', 'rating' => '4', 'sociability' => 'friendly'],
            ['attitude' => 'mean', 'class' => 'monster', 'rating' => '-1.5', 'sociability' => 'loner'],
            ['attitude' => 'nice', 'class' => 'not monster', 'rating' => '2.6', 'sociability' => 'friendly'],
            ['attitude' => 'mean', 'class' => 'monster', 'rating' => '-1', 'sociability' => 'friendly'],
            ['attitude' => 'nice', 'class' => 'not monster', 'rating' => '2.9', 'sociability' => 'friendly'],
            ['attitude' => 'nice', 'class' => 'not monster', 'rating' => '-5', 'sociability' => 'loner'],
        ];

        $records = iterator_to_array($this->extractor, false);

        $this->assertEquals($expected, $records);
    }
}
