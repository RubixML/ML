<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Extractors;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Extractors\ARFF;
use PHPUnit\Framework\TestCase;

use function file_put_contents;
use function is_float;
use function is_nan;
use function strtotime;
use function sys_get_temp_dir;
use function tempnam;
use function unlink;

#[Group('Extractors')]
#[CoversClass(ARFF::class)]
class ARFFTest extends TestCase
{
    protected ARFF $extractor;

    protected function setUp() : void
    {
        $this->extractor = new ARFF(
            path: 'tests/test.arff'
        );
    }

    #[Test]
    public function extractorRejectsInvalidPath() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new ARFF(path: 'tests');
    }

    #[Test]
    public function header() : void
    {
        $expected = [
            'attitude', 'texture', 'rating', 'visits', 'score', 'born', 'class label',
        ];

        $this->assertEquals($expected, $this->extractor->header());
    }

    #[Test]
    public function extract() : void
    {
        $expected = [
            [
                'attitude' => 'nice', 'texture' => 'furry', 'rating' => 4.0, 'visits' => 10,
                'score' => 0.5, 'born' => (float) strtotime('2016-01-01T12:00:00'),
                'class label' => 'not monster',
            ],
            [
                'attitude' => 'mean', 'texture' => 'furry', 'rating' => -1.5, 'visits' => '?',
                'score' => 0.9, 'born' => (float) strtotime('2015-06-15'),
                'class label' => 'monster',
            ],
            [
                'attitude' => 'nice', 'texture' => 'rough', 'rating' => 2.6, 'visits' => 20,
                'score' => 0.001, 'born' => (float) strtotime('2016-02-28'),
                'class label' => 'not monster',
            ],
            [
                'attitude' => 'mean', 'texture' => 'rough', 'rating' => -1.0, 'visits' => 30,
                'score' => 2.5, 'born' => (float) strtotime('2014-12-31T23:59:59'),
                'class label' => 'monster',
            ],
            [
                'attitude' => 'nice', 'texture' => 'rough', 'rating' => 2.9, 'visits' => 40,
                'score' => null, 'born' => (float) strtotime('2016-03-01'),
                'class label' => 'not monster',
            ],
        ];

        $records = iterator_to_array($this->extractor, false);

        $actual = array_map(function ($record) {
            return array_map(function ($value) {
                return (is_float($value) and is_nan($value)) ? null : $value;
            }, $record);
        }, $records);

        $this->assertEquals($expected, $actual);

        $this->assertSame(10, $records[0]['visits']);
        $this->assertSame('?', $records[1]['visits']);
        $this->assertTrue(is_nan($records[4]['score']));
    }

    #[Test]
    public function extractMultilineString() : void
    {
        $path = tempnam(sys_get_temp_dir(), 'arff_');

        file_put_contents($path, "@relation test\n@attribute text string\n@data\n'fur\nry'\n");

        $extractor = new ARFF($path);

        $expected = [
            ['text' => "fur\nry"],
        ];

        $this->assertEquals($expected, iterator_to_array($extractor, false));

        unlink($path);
    }

    #[Test]
    public function extractEscapedApostrophe() : void
    {
        $path = tempnam(sys_get_temp_dir(), 'arff_');

        file_put_contents(
            $path,
            "@relation test\n@attribute a numeric\n@attribute b string\n@data\n1,'don\\'t'\n2,'ok'\n"
        );

        $extractor = new ARFF($path);

        $expected = [
            ['a' => 1.0, 'b' => 'don\\\'t'],
            ['a' => 2.0, 'b' => 'ok'],
        ];

        $this->assertEquals($expected, iterator_to_array($extractor, false));

        unlink($path);
    }

    #[Test]
    public function extractEscapedApostropheWithComment() : void
    {
        $path = tempnam(sys_get_temp_dir(), 'arff_');

        file_put_contents(
            $path,
            "@relation test\n@attribute b string\n@data\n'don\\'t' % trailing note\n"
        );

        $extractor = new ARFF($path);

        $expected = [
            ['b' => 'don\\\'t'],
        ];

        $this->assertEquals($expected, iterator_to_array($extractor, false));

        unlink($path);
    }

    #[Test]
    public function attributeNameWithEscapedApostrophe() : void
    {
        $path = tempnam(sys_get_temp_dir(), 'arff_');

        file_put_contents(
            $path,
            "@relation test\n@attribute 'don\\'t' string\n@data\nhi\n"
        );

        $extractor = new ARFF($path);

        $this->assertEquals(["don't"], $extractor->header());

        $expected = [
            ["don't" => 'hi'],
        ];

        $this->assertEquals($expected, iterator_to_array($extractor, false));

        unlink($path);
    }

    #[Test]
    public function extractCustomPlaceholder() : void
    {
        $path = tempnam(sys_get_temp_dir(), 'arff_');

        file_put_contents($path, "@relation test\n@attribute born date\n@attribute class {monster,nice}\n@data\n?,?\n");

        $extractor = new ARFF($path, 'unknown');

        $expected = [
            ['born' => 'unknown', 'class' => 'unknown'],
        ];

        $this->assertEquals($expected, iterator_to_array($extractor, false));

        unlink($path);
    }

    #[Test]
    public function extractMissingValues() : void
    {
        $path = tempnam(sys_get_temp_dir(), 'arff_');

        file_put_contents($path, "@relation test\n@attribute born date\n@attribute class {monster,nice}\n@data\n?,?\n");

        $extractor = new ARFF($path);

        $expected = [
            ['born' => '?', 'class' => '?'],
        ];

        $this->assertEquals($expected, iterator_to_array($extractor, false));

        unlink($path);
    }

    #[Test]
    public function extractMalformedRecord() : void
    {
        $path = tempnam(sys_get_temp_dir(), 'arff_');

        file_put_contents($path, "@relation test\n@attribute a numeric\n@attribute b string\n@data\n1\n");

        $extractor = new ARFF($path);

        $this->expectException(RuntimeException::class);

        iterator_to_array($extractor, false);

        unlink($path);
    }

    #[Test]
    public function extractInvalidNumeric() : void
    {
        $path = tempnam(sys_get_temp_dir(), 'arff_');

        file_put_contents($path, "@relation test\n@attribute a numeric\n@data\nfoo\n");

        $extractor = new ARFF($path);

        $this->expectException(RuntimeException::class);

        iterator_to_array($extractor, false);

        unlink($path);
    }
}
