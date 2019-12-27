<?php

namespace Rubix\ML\Tests\Extractors;

use Rubix\ML\Extractors\NDJSON;
use PHPUnit\Framework\TestCase;
use IteratorAggregate;
use Traversable;

class NDJSONTest extends TestCase
{
    /**
     * @var \Rubix\ML\Extractors\NDJSON;
     */
    protected $extractor;

    public function setUp() : void
    {
        $this->extractor = new NDJSON('tests/test.ndjson');
    }

    public function test_build_factory() : void
    {
        $this->assertInstanceOf(NDJSON::class, $this->extractor);
        $this->assertInstanceOf(IteratorAggregate::class, $this->extractor);
        $this->assertInstanceOf(Traversable::class, $this->extractor);
    }

    public function test_extract() : void
    {
        $expected = [
            ['attitude' => 'nice', 'texture' => 'furry', 'sociability' => 'friendly', 'rating' => 4, 'class' => 'not monster'],
            ['attitude' => 'mean', 'texture' => 'furry', 'sociability' => 'loner', 'rating' => -1.5, 'class' => 'monster'],
            ['nice', 'rough', 'friendly', 2.6, 'not monster'],
            ['mean', 'rough', 'friendly', -1, 'monster'],
            ['nice', 'rough', 'friendly', 2.9, 'not monster'],
            ['nice', 'furry', 'loner', -5, 'not monster'],
        ];

        $records = iterator_to_array($this->extractor);

        $this->assertEquals($expected, array_values($records));
    }
}
