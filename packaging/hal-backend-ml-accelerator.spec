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


%prep
%setup -q

%build
%cmake \
  -DCMAKE_HAL_LIBDIR_PREFIX=%{_hal_libdir} \
  -DCMAKE_HAL_LICENSEDIR_PREFIX=%{_hal_licensedir} \
  %{?enable_dummy} \
  %{?enable_vivante} \
  %{?enable_snpe} \
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
