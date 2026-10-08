<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Extractors;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Extractors\Repeater;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Exceptions\InvalidArgumentException;
use PHPUnit\Framework\TestCase;
use ArrayObject;
use Generator;

use function array_merge;
use function count;
use function iterator_to_array;

#[Group('Extractors')]
#[CoversClass(Repeater::class)]
class RepeaterTest extends TestCase
{
    protected array $records;

    protected function setUp() : void
    {
        $this->records = [
            ['attitude' => 'nice', 'texture' => 'furry', 'sociability' => 'friendly', 'rating' => '4', 'class' => 'not monster'],
            ['attitude' => 'mean', 'texture' => 'furry', 'sociability' => 'loner', 'rating' => '-1.5', 'class' => 'monster'],
            ['attitude' => 'nice', 'texture' => 'rough', 'sociability' => 'friendly', 'rating' => '2.6', 'class' => 'not monster'],
            ['attitude' => 'mean', 'texture' => 'rough', 'sociability' => 'friendly', 'rating' => '-1', 'class' => 'monster'],
            ['attitude' => 'nice', 'texture' => 'rough', 'sociability' => 'friendly', 'rating' => '2.9', 'class' => 'not monster'],
            ['attitude' => 'nice', 'texture' => 'furry', 'sociability' => 'loner', 'rating' => '-5', 'class' => 'not monster'],
        ];
    }

    #[Test]
    public function constructWithInvalidRepetitions() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new Repeater($this->records, 0);
    }

    #[Test]
    public function rejectRawGeneratorWhenRepeating() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new Repeater($this->generate(), 3);
    }

    #[Test]
    public function repeatFromGeneratorTwice() : void
    {
        $extractor = new Repeater($this->generate(...), 2);

        $records = iterator_to_array($extractor, false);

        $this->assertSame(array_merge($this->records, $this->records), $records);
    }

    #[Test]
    public function factoryMustReturnIterable() : void
    {
        $extractor = new Repeater($this->invalidFactory(...), 2);

        $this->expectException(InvalidArgumentException::class);

        iterator_to_array($extractor, false);
    }

    #[Test]
    public function repeat() : void
    {
        $extractor = new Repeater($this->records, 3);

        $records = iterator_to_array($extractor, false);

        $expected = array_merge($this->records, $this->records, $this->records);

        $this->assertCount(3 * count($this->records), $records);

        $this->assertSame($expected, $records);
    }

    #[Test]
    public function repeatFromIteratorAggregate() : void
    {
        $extractor = new Repeater(new ArrayObject($this->records), 3);

        $records = iterator_to_array($extractor, false);

        $expected = array_merge($this->records, $this->records, $this->records);

        $this->assertCount(3 * count($this->records), $records);

        $this->assertSame($expected, $records);
    }

    #[Test]
    public function repeatFromCSV() : void
    {
        $extractor = new Repeater(new CSV(path: 'tests/test.csv', header: true), 3);

        $records = iterator_to_array($extractor, false);

        $expected = array_merge($this->records, $this->records, $this->records);

        $this->assertCount(3 * count($this->records), $records);

        $this->assertSame($expected, $records);
    }

    #[Test]
    public function repeatFromCSVIsReiterable() : void
    {
        $extractor = new Repeater(new CSV(path: 'tests/test.csv', header: true), 3);

        $expected = array_merge($this->records, $this->records, $this->records);

        $firstPass = iterator_to_array($extractor, false);

        $secondPass = iterator_to_array($extractor, false);

        $this->assertSame($expected, $firstPass);

        $this->assertSame($expected, $secondPass);
    }

    /**
     * @return Generator<mixed[]>
     */
    protected function generate() : Generator
    {
        yield from $this->records;
    }

    protected function invalidFactory() : int
    {
        return 42;
    }
}
