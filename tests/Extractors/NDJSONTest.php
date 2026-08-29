<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Extractors;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Extractors\NDJSON;
use PHPUnit\Framework\TestCase;

#[Group('Extractors')]
#[CoversClass(NDJSON::class)]
class NDJSONTest extends TestCase
{
    protected NDJSON $extractor;

    protected function setUp() : void
    {
        $this->extractor = new NDJSON('tests/test.ndjson');
    }

    public function testExtractExport() : void
    {
        $expected = [
            ['attitude' => 'nice', 'texture' => 'furry', 'sociability' => 'friendly', 'rating' => 4, 'class' => 'not monster'],
            ['attitude' => 'mean', 'texture' => 'furry', 'sociability' => 'loner', 'rating' => -1.5, 'class' => 'monster'],
            ['attitude' => 'nice', 'texture' => 'rough', 'sociability' => 'friendly', 'rating' => 2.6, 'class' => 'not monster'],
            ['attitude' => 'mean', 'texture' => 'rough', 'sociability' => 'friendly', 'rating' => -1, 'class' => 'monster'],
            ['attitude' => 'nice', 'texture' => 'rough', 'sociability' => 'friendly', 'rating' => 2.9, 'class' => 'not monster'],
            ['attitude' => 'nice', 'texture' => 'furry', 'sociability' => 'loner', 'rating' => -5, 'class' => 'not monster'],
        ];

        $records = iterator_to_array($this->extractor, false);

        $this->assertEquals($expected, array_values($records));

        $this->extractor->export($records, true);

        $this->assertFileExists('tests/test.ndjson');
    }
}
