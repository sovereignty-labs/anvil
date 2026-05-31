package runtime

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"os"
	"path/filepath"
	"testing"
)

func TestExtractRuntimePreservesTarGzSymlinks(t *testing.T) {
	dir := t.TempDir()
	archivePath := filepath.Join(dir, "runtime.tar.gz")
	destDir := filepath.Join(dir, "out")

	f, err := os.Create(archivePath)
	if err != nil {
		t.Fatal(err)
	}

	gzw := gzip.NewWriter(f)
	tw := tar.NewWriter(gzw)

	writeEntry := func(hdr *tar.Header, content []byte) {
		t.Helper()
		if err := tw.WriteHeader(hdr); err != nil {
			t.Fatal(err)
		}
		if len(content) > 0 {
			if _, err := tw.Write(content); err != nil {
				t.Fatal(err)
			}
		}
	}

	writeEntry(&tar.Header{
		Name:     "bundle/libfoo.so.0.0.0",
		Mode:     0o755,
		Size:     int64(len("ELFCONTENT")),
		Typeflag: tar.TypeReg,
	}, []byte("ELFCONTENT"))
	writeEntry(&tar.Header{
		Name:     "bundle/libfoo.so.0",
		Mode:     0o777,
		Typeflag: tar.TypeSymlink,
		Linkname: "libfoo.so.0.0.0",
	}, nil)
	writeEntry(&tar.Header{
		Name:     "bundle/libfoo.so",
		Mode:     0o777,
		Typeflag: tar.TypeSymlink,
		Linkname: "libfoo.so.0",
	}, nil)

	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := gzw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	if err := extractRuntime(archivePath, destDir); err != nil {
		t.Fatalf("extractRuntime: %v", err)
	}

	fi, err := os.Lstat(filepath.Join(destDir, "libfoo.so.0"))
	if err != nil {
		t.Fatalf("lstat libfoo.so.0: %v", err)
	}
	if fi.Mode()&os.ModeSymlink == 0 {
		t.Fatalf("libfoo.so.0 mode = %v, want symlink", fi.Mode())
	}
	target, err := os.Readlink(filepath.Join(destDir, "libfoo.so.0"))
	if err != nil {
		t.Fatalf("readlink libfoo.so.0: %v", err)
	}
	if target != "libfoo.so.0.0.0" {
		t.Fatalf("libfoo.so.0 target = %q, want libfoo.so.0.0.0", target)
	}

	fi, err = os.Lstat(filepath.Join(destDir, "libfoo.so"))
	if err != nil {
		t.Fatalf("lstat libfoo.so: %v", err)
	}
	if fi.Mode()&os.ModeSymlink == 0 {
		t.Fatalf("libfoo.so mode = %v, want symlink", fi.Mode())
	}

	data, err := os.ReadFile(filepath.Join(destDir, "libfoo.so"))
	if err != nil {
		t.Fatalf("readfile libfoo.so: %v", err)
	}
	if !bytes.Equal(data, []byte("ELFCONTENT")) {
		t.Fatalf("libfoo.so contents = %q, want ELFCONTENT", data)
	}

	fi, err = os.Lstat(filepath.Join(destDir, "libfoo.so.0.0.0"))
	if err != nil {
		t.Fatalf("lstat libfoo.so.0.0.0: %v", err)
	}
	if fi.Mode()&os.ModeSymlink != 0 {
		t.Fatalf("libfoo.so.0.0.0 mode = %v, want regular file", fi.Mode())
	}
}
