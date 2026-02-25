Name:       hal-backend-ml-accelerator
Summary:    ML HAL backend drivers for various targets
# Synchronize the version information.
# 1. CMake : ./CMakeLists.txt
# 2. Tizen : ./packaging/hal-backend-ml-accelerator.spec
Version:    0.1.0
Release:    0
Group:      Machine Learning/ML Framework
License:    Apache-2.0
Source0:    %{name}-%{version}.tar.gz

%define _module_name      hal-backend-ml-accelerator
%define _module_name_snpe     hal-backend-ml-snpe
%define _module_name_vivante   hal-backend-ml-vivante
%define _module_name_dummypassthrough   hal-backend-ml-dummy-passthrough
BuildRequires:  cmake
BuildRequires:  pkgconfig(hal-rootstrap)

# For DA
%if 0%{?_with_da_profile}

## For meson board
%if 0%{?_with_meson64}
%define         vivante_support 1
%endif

## For qrb board
%if 0%{?_with_qrb4210}
%define         snpe_support 1
%endif

%endif # For DA

%if 0%{?build_tests}
BuildRequires: gtest-devel
%endif

%description
ML HAL backend drivers for various targets

# Config dummy backend (dummy-passthrough)
%define         dummy_support 1

%if 0%{?dummy_support}
%package dummy
Summary:  dummy backend for hal-backend-ml-accelerator
%description dummy
%define enable_dummy -DENABLE_DUMMY=ON
%endif

# Config vivante
%if 0%{?vivante_support}
%package vivante
Summary:  hal-backend-ml-accelerator for vivante
%description vivante
%define enable_vivante -DENABLE_VIVANTE=ON
%endif

# Config snpe
%if 0%{?snpe_support}
%package snpe
Summary:  hal-backend-ml-accelerator for snpe
%description snpe
%define enable_snpe -DENABLE_SNPE=ON
%endif

%if 0%{?build_tests}
%define _testdir %{_hal_bindir}/ml-accelerator/
%define enable_tests -DBUILD_TESTS=ON -DTEST_DIR=%{_testdir}

%package halbackendtest
Summary:    Test Binary for Hal backend
Requires: %{name} = %{version}-%{release}
%description halbackendtest
Test Binary for hal-backend

%files halbackendtest
%manifest packaging/hal-backend-ml-accelerator.manifest
%if 0%{?dummy_support}
%{_testdir}%{_module_name_dummypassthrough}-test
%endif
%if 0%{?vivante_support}
%{_testdir}%{_module_name_vivante}-test
%endif
%if 0%{?snpe_support}
%{_testdir}%{_module_name_snpe}-test
%endif
%else
%define enable_tests -DBUILD_TESTS=OFF
%endif

%prep
%setup -q

%build
%cmake \
  -DCMAKE_HAL_LIBDIR_PREFIX=%{_hal_libdir} \
  -DCMAKE_HAL_LICENSEDIR_PREFIX=%{_hal_licensedir} \
  %{?enable_dummy} \
  %{?enable_vivante} \
  %{?enable_snpe} \
  %{?enable_tests} \
  .
make %{?_smp_mflags}

%install
%make_install

%post
/sbin/ldconfig

%postun
/sbin/ldconfig

%if 0%{?dummy_support}
%files dummy
%manifest packaging/hal-backend-ml-accelerator.manifest
%license LICENSE
%{_hal_libdir}/libhal-backend-ml-dummy-passthrough.so
%endif

%if 0%{?vivante_support}
%files vivante
%manifest packaging/hal-backend-ml-accelerator.manifest
%license LICENSE
%{_hal_libdir}/libhal-backend-ml-vivante.so
%endif

%if 0%{?snpe_support}
%files snpe
%manifest packaging/hal-backend-ml-accelerator.manifest
%license LICENSE
%{_hal_libdir}/libhal-backend-ml-snpe.so
%endif

%changelog
* Wed Aug 27 2025 Yongjoo Ahn <yongjoo1.ahn@samsung.com>
- Release of 0.1.0 (Tizen 10.0 M2)
