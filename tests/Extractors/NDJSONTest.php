<?php

namespace Rubix\ML\Tests\Extractors;

use Rubix\ML\Extractors\NDJSON;
use Rubix\ML\Extractors\Extractor;
use PHPUnit\Framework\TestCase;
use IteratorAggregate;

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
        $this->assertInstanceOf(Extractor::class, $this->extractor);
        $this->assertInstanceOf(IteratorAggregate::class, $this->extractor);
    }

    public function test_extract() : void
    {
        $records = $this->extractor->setOffset(1)->extract();

        $expected = [
            // ['attitude' => 'nice', 'appearance' => 'furry', 'sociability' => 'friendly', 'rating' => 4, 'class' => 'not monster'],
            ['attitude' => 'mean', 'appearance' => 'furry', 'sociability' => 'loner', 'rating' => -1.5, 'class' => 'monster'],
            ['nice', 'rough', 'friendly', 2.6, 'not monster'],
            ['mean', 'rough', 'friendly', -1, 'monster'],
            ['nice', 'rough', 'friendly', 2.9, 'not monster'],
            ['nice', 'furry', 'loner', -5, 'not monster'],
        ];

        $records = is_array($records) ? $records : iterator_to_array($records);

        $this->assertEquals($expected, array_values($records));
    }
}
